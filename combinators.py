#!/usr/bin/env python3

import collections

import probtorch
from probtorch.stochastic import Provenance
import pygtrie
import torch
import torch.nn as nn

import traces
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
        empty_trace = traces.Traces(traces={self.name: probtorch.Trace()})
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
        ps = traces.Traces(traces={self.name: self.p})
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

class SideEffect(ModelSampler):
    def __init__(self, first, second):
        super(SideEffect, self).__init__()
        assert isinstance(first, Sampler)
        assert isinstance(second, Sampler)
        self.add_module('first', first)
        self.add_module('second', second)

    @property
    def name(self):
        return 'SideEffect(%s, %s)' % (self.first.name, self.second.name)

    def _forward(self, *args, **kwargs):
        _, kwargs['trace'] = self.first(*args, **kwargs)
        return self.second(*args, **kwargs)

def score_under_proposal(proposal, model):
    return SideEffect(proposal, Score(model))

class Lambda(ModelSampler):
    def __init__(self, body, nargs=0, kwargs=[]):
        super(Lambda, self).__init__()
        assert isinstance(body, Sampler)
        self.add_module('body', body)
        self._nargs = nargs
        assert 'trace' not in kwargs
        self._kwargs = kwargs

    @property
    def name(self):
        return 'Lambda(%s, %d, %s)' % (self.body.name, self._nargs, self._kwargs)

    def _forward(self, *args, **kwargs):
        args = list(args)
        for _ in range(self._nargs):
            args = args.pop(0)
        args = tuple(args)
        for k in self._kwargs:
            kwargs.pop(k)
        return self.body(*args, **kwargs)

class Composition(ModelSampler):
    def __init__(self, outer, inner, intermediate_name=None):
        super(Composition, self).__init__()
        assert isinstance(outer, Sampler)
        assert isinstance(inner, Sampler)
        self.add_module('outer', outer)
        self.add_module('inner', inner)
        self._intermediate = intermediate_name

    @property
    def name(self):
        return self.outer.name + '.' + self.inner.name

    def _forward(self, *args, **kwargs):
        final_trace = kwargs.pop('trace')
        kwargs['trace'] = final_trace.extract(self.inner.name)
        temp, inner_trace = self.inner(*args, **kwargs)
        kws = {'trace': final_trace.extract(self.outer.name)}
        if self._intermediate:
            kws[self._intermediate] = temp
            result, outer_trace = self.outer(**kws)
        elif isinstance(temp, tuple):
            result, outer_trace = self.outer(*temp, **kws)
        else:
            result, outer_trace = self.outer(temp, **kws)
        final_trace.insert(self.inner.name, inner_trace)
        final_trace.insert(self.outer.name, outer_trace)
        return result, final_trace

class Partial(ModelSampler):
    def __init__(self, func, *arguments, **keywords):
        super(Partial, self).__init__()
        self.add_module('curried', func)
        self._curry_arguments = arguments
        self._curry_kwargs = keywords

    @property
    def name(self):
        return 'Partial(%s)' % self.curried.name

    def _forward(self, *args, **kwargs):
        kwargs = {**kwargs, **self._curry_kwargs}
        return self.curried(*self._curry_arguments, *args, **kwargs)

class MapIid(ModelSampler):
    def __init__(self, func, items, **kwargs):
        super(MapIid, self).__init__()
        assert isinstance(func, Sampler)
        self.add_module('func', func)
        self.map_items = items
        self.map_kwargs = kwargs

    @property
    def name(self):
        return 'MapIid(%s)' % self.func.name

    def iterate(self, trace, **kwargs):
        for item in self.map_items:
            kwargs = {**self.map_kwargs, **kwargs}
            kwargs['trace'] = trace.extract(self.map_func.name + str(item))
            result, step_trace = self.map_func(item, **kwargs)
            trace.insert(self.map_func.name + str(item), step_trace)
            yield result

    def _forward(self, *args, **kwargs):
        trace = kwargs.pop('trace')
        result = list(self.iterate(trace, **kwargs))
        return result, trace

class Population(InferenceSampler):
    def __init__(self, sampler, particle_shape, before=True):
        super(Population, self).__init__(sampler)
        self._particle_shape = particle_shape
        self._before = before

    @property
    def particle_shape(self):
        return self._particle_shape

    @property
    def before(self):
        return self._before

    @property
    def expander(self):
        return lambda v: utils.batch_expand(v, self.particle_shape)

    def _expand_args(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = utils.batch_expand(arg, self.particle_shape)
            elif isinstance(arg, collections.Mapping):
                args[i] = utils.vardict_map(arg, self.expander)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = utils.batch_expand(v, self.particle_shape)
            elif isinstance(v, collections.Mapping):
                kwargs[k] = utils.vardict_map(v, self.expander)
        return tuple(args), kwargs

    def sample_prehook(self, trace, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        return trace, args, kwargs

    def sample_hook(self, results, trace):
        if not self.before:
            results = self._expand_args(*results)
        return results, trace

def hyper_population(sampler, particle_shape, trainable={}, hyper={}):
    return ParamCall(Population(sampler, particle_shape, before=True),
                     trainable=trainable, hyper=hyper)
