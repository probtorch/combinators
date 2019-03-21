#!/usr/bin/env python3

from contextlib import contextmanager
import torch

import probtorch
from probtorch.stochastic import Provenance

from .. import graphs
from ..sampler import Sampler
from .. import utils

class Model(Sampler):
    def __init__(self, batch_shape=(1,)):
        super(Model, self).__init__()
        self._batch_shape = batch_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def name(self):
        raise NotImplementedError()

    def get_model(self):
        return self

    @property
    def arguments(self):
        return ()

class Deterministic(Model):
    def __init__(self, *args, batch_shape=(1,)):
        super(Deterministic, self).__init__(batch_shape=batch_shape)
        self._args, _ = self._expand_args(*args)

    @property
    def name(self):
        return 'Deterministic'

    def forward(self, *args, **kwargs):
        empty_graph = graphs.ComputationGraph(traces={self.name: probtorch.Trace()})
        return self._args, empty_graph, torch.zeros(self.batch_shape)

    @contextmanager
    def cond(self, qs):
        yield self

    def walk(self, f):
        return f(self)

def deterministic(*args, batch_shape=(1,)):
    return Deterministic(*args, batch_shape=batch_shape)

class Primitive(Model):
    def __init__(self, params={}, trainable=True, batch_shape=(1,), q=None):
        super(Primitive, self).__init__(batch_shape=batch_shape)
        self._hyperparams_trainable = trainable
        self.register_args(params, trainable)
        self.p = None
        self.q = q if q else probtorch.Trace()

    @property
    def hyperparams_trainable(self):
        return self._hyperparams_trainable

    @property
    def name(self):
        return self.__class__.__name__

    def sample(self, Dist, *args, name=None, value=None, **kwargs):
        if name in self.q:
            assert value is None or (value == self.q[name].value).all()
            value = self.q[name].value
        if value is not None:
            if name in self.q and not self.q[name].observed:
                if self.q[name].provenance == Provenance.RESCORE:
                    provenance = Provenance.SAMPLED
                else:
                    provenance = Provenance.REUSED
            else:
                provenance = Provenance.OBSERVED
                shared_shapes = utils.broadcastable_sizes(value.shape,
                                                          self.batch_shape)
                if shared_shapes != self.batch_shape:
                    value = value.expand(*self.batch_shape, *value.shape)
        else:
            provenance = Provenance.SAMPLED
        return self.p.variable(Dist, *args, **kwargs, name=name, value=value,
                               provenance=provenance)

    def param_sample(self, Dist, name):
        params = self.args_vardict()[name]
        for arg, val in params.items():
            matches = [k for k in utils.PARAM_TRANSFORMS if k in arg]
            if matches:
                params[arg] = utils.PARAM_TRANSFORMS[matches[0]](val)
        return self.sample(Dist, name=name, **params)

    def observe(self, name, value, Dist, *args, **kwargs):
        assert name not in self.q or self.q[name].observed
        for arg in args:
            if isinstance(arg, torch.Tensor):
                value = value.to(device=arg.device)
                break
        for kwarg in kwargs.values():
            if isinstance(kwarg, torch.Tensor):
                value = value.to(device=kwarg.device)
                break
        return self.sample(Dist, *args, name=name, value=value, **kwargs)

    def param_observe(self, Dist, name, value):
        params = self.args_vardict()[name]
        for arg, val in params.items():
            matches = [k for k in utils.PARAM_TRANSFORMS if k in arg]
            if matches:
                params[arg] = utils.PARAM_TRANSFORMS[matches[0]](val)
        return self.observe(name, value, Dist, **params)

    def factor(self, log_prob, name=None):
        assert name not in self.q or isinstance(self.q[name], probtorch.Factor)
        return self.p.factor(log_prob, name=name)

    def loss(self, objective, value, target, name=None):
        assert name not in self.q or isinstance(self.q[name], probtorch.Loss)
        return self.p.loss(objective, value, target, name=name)

    def walk(self, f):
        return f(self)

    @contextmanager
    def cond(self, qs):
        q = self.q
        try:
            self.q = qs[self.name]
            yield self
        finally:
            self.q = q

    def forward(self, *args, **kwargs):
        self.p = probtorch.Trace()
        result = self._forward(*args, **kwargs)
        ps = graphs.ComputationGraph(traces={self.name: self.p})
        reused = [k for k in self.p if k in self.q and\
                  utils.reused_variable(self.p[k])]
        log_weight = torch.zeros(self.batch_shape,
                                 device=ps.device)
        conditioned = [k for k in self.p.conditioned()]
        sample_dims = tuple(range(len(self.batch_shape)))
        log_weight += self.p.log_joint(sample_dims=sample_dims,
                                       nodes=conditioned,
                                       reparameterized=False) +\
                      self.p.log_joint(sample_dims=sample_dims, nodes=reused,
                                       reparameterized=False)
        self.p = None
        assert log_weight.shape == self.batch_shape
        return result, ps, log_weight

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

class Compose(Model):
    def __init__(self, f, g, kw=None):
        assert isinstance(f, Sampler)
        assert isinstance(g, Sampler)
        assert f.batch_shape == g.batch_shape
        super(Compose, self).__init__(batch_shape=f.batch_shape)
        self.add_module('f', f)
        self.add_module('g', g)
        self._kw = kw

    @property
    def name(self):
        return '%s.%s' % (self.f.name, self.g.name)

    def forward(self, *args, **kwargs):
        zg, xi_g, w_g = self.g(*args, **kwargs)
        kws = {}
        if self._kw:
            kws[self._kw] = zg
        elif not isinstance(zg, tuple):
            zg = (zg,)
        zf, xi_f, w_f = self.f(*zg, **kws)
        xi = graphs.ComputationGraph()
        xi.insert(self.name, xi_g)
        xi.insert(self.name, xi_f)
        return zf, xi, w_g + w_f

    @contextmanager
    def cond(self, qs):
        with self.f.cond(qs[self.name:]) as fq:
            with self.g.cond(qs[self.name:]) as gq:
                yield self

    def walk(self, f):
        walk_f = self.f.walk(f)
        walk_g = self.g.walk(f)
        return f(Compose(walk_f, walk_g, self._kw))

def compose(f, g, kw=None):
    return Compose(f, g, kw=kw)

class Partial(Model):
    def __init__(self, func, *arguments, **keywords):
        assert isinstance(func, Sampler)
        super(Partial, self).__init__(batch_shape=func.batch_shape)
        self.add_module('curried', func)
        self._curry_arguments = arguments
        self._curry_kwargs = keywords

    @property
    def name(self):
        return 'Partial'

    def forward(self, *args, **kwargs):
        return self.curried(*self._curry_arguments, *args, **kwargs,
                            **self._curry_kwargs)

    def walk(self, f):
        walk_curried = self.curried.walk(f)
        return f(Partial(walk_curried, *self._curry_arguments,
                         **self._curry_kwargs))

    @contextmanager
    def cond(self, qs):
        with self.curried.cond(qs[self.name + '/' + self.curried.name:]) as cq:
            yield self

def partial(func, *arguments, **keywords):
    return Partial(func, *arguments, **keywords)

class MapIid(Model):
    def __init__(self, func):
        assert isinstance(func, Sampler)
        super(MapIid, self).__init__(batch_shape=func.batch_shape)
        self.add_module('func', func)

    @property
    def name(self):
        return 'MapIid'

    def iterate(self, items, *args, **kwargs):
        for item in items:
            yield self.func(item, *args, **kwargs)

    def forward(self, items, *args, **kwargs):
        results = list(self.iterate(items, *args, **kwargs))
        graph = graphs.ComputationGraph()
        log_weight = torch.zeros(self.batch_shape)
        for (i, (_, xi, w)) in enumerate(results):
            graph.insert(self.name + '/%d' % i, xi)
            log_weight += w
        zs = [result[0] for result in results]
        return zs, graph, log_weight

    def walk(self, f):
        return f(MapIid(self.func.walk(f)))

    @contextmanager
    def cond(self, qs):
        with self.func.cond(qs['/' + self.func.name:]) as funcq:
            yield self

def map_iid(func):
    return MapIid(func)
