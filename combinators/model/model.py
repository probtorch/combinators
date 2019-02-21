#!/usr/bin/env python3

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

class Deterministic(Model):
    def __init__(self, *args, batch_shape=(1,)):
        super(Deterministic, self).__init__(batch_shape=batch_shape)
        self._args, _ = self._expand_args(*args)

    @property
    def name(self):
        return 'Deterministic'

    def forward(self, *args, **kwargs):
        empty_graph = graphs.ModelGraph(traces={self.name: probtorch.Trace()})
        return self._args, empty_graph, torch.zeros(self.batch_shape)

    def cond(self, qs):
        return Deterministic(*self._args)

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
    def name(self):
        return self.__class__.__name__

    def sample(self, Dist, *args, name=None, value=None, **kwargs):
        if name in self.q:
            assert value is None or (value == self.q[name].value).all()
            value = self.q[name].value
        if value is not None:
            if name in self.q and not self.q[name].observed:
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
        return self.sample(Dist, *args, name=name, value=value, **kwargs)

    def walk(self, f):
        return f(self)

    def cond(self, qs):
        return self.__class__(params=self.args_vardict(False),
                              trainable=self._hyperparams_trainable,
                              batch_shape=self.batch_shape, q=qs[self.name])

    def forward(self, *args, **kwargs):
        self.p = probtorch.Trace()
        result = self._forward(*args, **kwargs)
        priors = [k for k in self.p if k in self.q]
        log_weight = torch.zeros(self.batch_shape)
        likelihoods = [k for k in self.p if k in self.p.conditioned() and\
                       k not in self.q]
        sample_dims = tuple(range(len(self.batch_shape)))
        log_weight += self.p.log_joint(sample_dims=sample_dims,
                                       nodes=likelihoods,
                                       reparameterized=False) +\
                      self.p.log_joint(sample_dims=sample_dims, nodes=priors,
                                       reparameterized=False) -\
                      self.q.log_joint(sample_dims=sample_dims, nodes=priors,
                                       reparameterized=False)
        ps = graphs.ModelGraph(traces={self.name: self.p})
        self.p = None
        assert log_weight.shape == self.batch_shape
        return result, ps, log_weight

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

class Compose(Model):
    def __init__(self, f, g, intermediate_name=None):
        assert isinstance(f, Sampler)
        assert isinstance(g, Sampler)
        assert f.batch_shape == g.batch_shape
        super(Compose, self).__init__(batch_shape=f.batch_shape)
        self.add_module('f', f)
        self.add_module('g', g)
        self._intermediate = intermediate_name

    @property
    def name(self):
        return '%s.%s' % (self.f.name, self.g.name)

    def forward(self, *args, **kwargs):
        zg, xi_g, w_g = self.g(*args, **kwargs)
        kws = {}
        if self._intermediate:
            kws[self._intermediate] = zg
        elif not isinstance(zg, tuple):
            zg = (zg,)
        zf, xi_f, w_f = self.f(*zg, **kws)
        xi = graphs.ModelGraph()
        xi.insert(self.name, xi_g)
        xi.insert(self.name, xi_f)
        return zf, xi, w_g + w_f

    def cond(self, qs):
        fq = self.f.cond(qs[self.name:])
        gq = self.g.cond(qs[self.name:])
        return Compose(fq, gq, self._intermediate)

    def walk(self, f):
        walk_f = self.f.walk(f)
        walk_g = self.g.walk(f)
        return f(Compose(walk_f, walk_g, self._intermediate))

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

    def cond(self, qs):
        curried_q = self.curried.cond(qs[self.name + '/' + self.curried.name:])
        return Partial(curried_q, *self._curry_arguments, **self._curry_kwargs)

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
        graph = graphs.ModelGraph()
        log_weight = torch.zeros(self.batch_shape)
        for (i, (_, xi, w)) in enumerate(results):
            graph.insert(self.name + '/%d' % i, xi)
            log_weight += w
        zs = [result[0] for result in results]
        return zs, graph, log_weight

    def walk(self, f):
        return f(MapIid(self.func.walk(f)))

    def cond(self, qs):
        funcq = self.func.cond(qs['/' + self.func.name:])
        return MapIid(funcq)
