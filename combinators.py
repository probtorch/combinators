#!/usr/bin/env python3

import collections

import probtorch
from probtorch.stochastic import Provenance
import pygtrie
import torch
import torch.nn as nn

import graphs
import utils

class Sampler(nn.Module):
    @property
    def name(self):
        raise NotImplementedError()

    @property
    def batch_shape(self):
        return torch.Size(self._batch_shape)

    def get_model(self):
        raise NotImplementedError()

    def walk(self, f):
        raise NotImplementedError()

    def cond(self, qs):
        raise NotImplementedError()

    @property
    def _expander(self):
        return lambda v: utils.batch_expand(v, self.batch_shape)

    def _expand_args(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = utils.batch_expand(arg, self.batch_shape)
            elif isinstance(arg, collections.Mapping):
                args[i] = utils.vardict_map(arg, self._expander)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = utils.batch_expand(v, self.batch_shape)
            elif isinstance(v, collections.Mapping):
                kwargs[k] = utils.vardict_map(v, self._expander)
        return tuple(args), kwargs

    def _args_vardict(self, expand=True):
        result = utils.vardict(self.state_dict(keep_vars=True))
        result = utils.vardict({k: v for k, v in result.items()
                                if '.' not in k})
        # PyTorch BUG: Parameter's don't get counted as Tensors in Normal
        for k, v in result.items():
            result[k] = v.clone()
        if expand:
            (result,), _ = self._expand_args(result)
        return result

    def args_vardict(self, expand=True):
        result = utils.vardict({})
        for module in self.children():
            if isinstance(module, Sampler):
                args = module.args_vardict(expand=expand)
                for k, v in args.items():
                    result[k] = v
        args = self._args_vardict(expand=expand)
        for k, v in args.items():
            result[k] = v
        return result

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            v = v.clone().detach()
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

class Model(Sampler):
    def __init__(self, batch_shape=(1,)):
        super(Model, self).__init__()
        self._batch_shape = batch_shape

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
        empty_trace = graphs.ModelGraph(traces={self.name: probtorch.Trace()})
        return self._args, empty_trace, torch.zeros(self.batch_shape)

    def cond(self, qs):
        return Deterministic(*self._args)

    def walk(self, f):
        return f(self)

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
            if name in self.q:
                provenance = Provenance.REUSED
            else:
                provenance = Provenance.OBSERVED
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

class Inference(Sampler):
    def __init__(self, sampler):
        super(Inference, self).__init__()
        assert isinstance(sampler, Sampler)
        self.add_module('sampler', sampler)

    @property
    def name(self):
        return self.get_model().name

    @property
    def batch_shape(self):
        return self.sampler.batch_shape

    def get_model(self):
        return self.sampler.get_model()

    def walk(self, f):
        raise NotImplementedError()

class GuidedConditioning(Inference):
    def __init__(self, sampler, guide):
        super(GuidedConditioning, self).__init__(sampler)
        assert isinstance(guide, Sampler)
        assert guide.batch_shape == sampler.batch_shape
        self.add_module('guide', guide)

    def forward(self, *args, **kwargs):
        _, xi, w = self.guide(*args, **kwargs)
        return self.sampler.cond(xi)(*args, **kwargs)

    def walk(self, f):
        return f(GuidedConditioning(self.sampler.walk(f), self.guide))

    def cond(self, qs):
        return GuidedConditioning(self.sampler.cond(qs), self.guide)

class Composition(Model):
    def __init__(self, f, g, intermediate_name=None):
        assert isinstance(f, Sampler)
        assert isinstance(g, Sampler)
        assert f.batch_shape == g.batch_shape
        super(Composition, self).__init__(batch_shape=f.batch_shape)
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
        xi = traces.Traces()
        xi.insert(self.name, xi_g)
        xi.insert(self.name, xi_f)
        return zf, xi, w_g + w_f

    def cond(self, qs):
        fq = self.f.cond(qs[self.name:])
        gq = self.g.cond(qs[self.name:])
        return Composition(fq, gq, self._intermediate)

    def walk(self, f):
        walk_f = self.f.walk(f)
        walk_g = self.g.walk(f)
        return f(Composition(walk_f, walk_g, self._intermediate))

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
    def __init__(self, func, items, **kwargs):
        assert isinstance(func, Sampler)
        super(MapIid, self).__init__(batch_shape=func.batch_shape)
        self.add_module('func', func)
        self.map_items = items
        self.map_kwargs = kwargs

    @property
    def name(self):
        return 'MapIid'

    def iterate(self, **kwargs):
        for item in self.map_items:
            kwargs = {**self.map_kwargs, **kwargs}
            z, xi, w = self.map_func(item, **kwargs)
            yield (z, xi, w)

    def forward(self, *args, **kwargs):
        results = list(self.iterate(**kwargs))
        trace = traces.Traces()
        weight = torch.zeros(self.batch_shape)
        for (_, xi, w) in results:
            trace.insert(self.name, xi)
            weight += w
        zs = [result[0] for result in results]
        return zs, trace, weight

    def walk(self, f):
        return f(MapIid(self.func.walk(f), self.map_items, **self.map_kwargs))

    def cond(self, qs):
        funcq = self.func.cond(qs['/' + self.func.name:])
        return MapIid(funcq, self.map_items, **self.map_kwargs)

class Population(Inference):
    def __init__(self, sampler, batch_shape, before=True):
        super(Population, self).__init__(sampler)
        self._batch_shape = batch_shape
        self._before = before

    @property
    def batch_shape(self):
        return self._particle_shape + self.sampler.batch_shape

    @property
    def before(self):
        return self._before

    def forward(self, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        z, xi, w = self.sampler(*args, **kwargs)
        if not isinstance(z, tuple):
            z = (z,)
        if not self.before:
            z = self._expand_args(*z)
        return z, xi, w

    def walk(self, f):
        return f(Population(self.sampler.walk(f), batch_shape=self._batch_shape,
                            before=self.before))

    def cond(self, qs):
        return Population(self.sampler.cond(qs), batch_shape=self._batch_shape,
                          before=self.before)
