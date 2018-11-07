#!/usr/bin/env python3

import collections
import functools
import inspect

import numpy as np
import probtorch
from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
import torch.nn as nn

import trace_tries
import utils

class Sampler(nn.Module):
    @property
    def name(self):
        raise NotImplementedError()

    def simulate(self, *args, **kwargs):
        result, trace = self.forward(*args, **kwargs)
        return trace.log_weight(), trace, result

class ModelSampler(Sampler):
    def forward(self, *args, **kwargs):
        if 'trace' not in kwargs:
            kwargs['trace'] = trace_tries.HierarchicalTrace()
        return self._forward(*args, **kwargs), kwargs['trace']

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

class PrimitiveCall(ModelSampler):
    def __init__(self, primitive, name=None, trainable={}, hyper={}):
        super(PrimitiveCall, self).__init__()
        self.register_args(trainable, True)
        self.register_args(hyper, False)
        assert not isinstance(primitive, Sampler)
        if isinstance(primitive, nn.Module):
            self.add_module('primitive', primitive)
            self._name = name
        else:
            self.primitive = primitive
            self._name = primitive.__name__

    @property
    def name(self):
        if isinstance(self.primitive, nn.Module):
            return self._name
        return self.primitive.__name__

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            v = torch.tensor(v)
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def args_vardict(self):
        result = utils.vardict(self.state_dict(keep_vars=True))
        # PyTorch BUG: Parameter's don't get counted as Tensors in Normal
        for k, v in result.items():
            result[k] = v.clone()
        return result

    def _forward(self, *args, **kwargs):
        params = self.args_vardict()
        if len(params):
            kwargs['params'] = params
        return self.primitive(*args, **kwargs)

class InferenceSampler(Sampler):
    def __init__(self, sampler):
        super(InferenceSampler, self).__init__()
        assert isinstance(sampler, Sampler)
        self.add_module('sampler', sampler)

    @property
    def name(self):
        return self.sampler.name

    def forward(self, *args, **kwargs):
        result, trace = self.sampler(*args, **kwargs)
        return self.infer(result, trace)

    def infer(self, results, trace):
        raise NotImplementedError()

class ProposalScore(InferenceSampler):
    def __init__(self, proposal, model):
        super(ProposalScore, self).__init__(model)
        assert isinstance(proposal, Sampler)
        self.add_module('proposal', proposal)

    @property
    def name(self):
        return self.sampler.name + '_' + self.proposal.name

    def forward(self, *args, **kwargs):
        _, trace = self.proposal(*args, **kwargs)
        kwargs['trace'] = trace_tries.HierarchicalTrace(proposal=trace)
        return super(ProposalScore, self).forward(*args, **kwargs)

    def infer(self, results, trace):
        return results, trace

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
        temp, trace = self._inner(*args, **kwargs)
        if self._intermediate:
            kws = {'trace': trace, self._intermediate: temp}
            return self._outer(**kws)
        return self._outer(*temp, trace=trace) if isinstance(temp, tuple) else\
               self._outer(temp, trace=trace)

class Partial(ModelSampler):
    def __init__(self, func, *arguments, **keywords):
        super(Partial, self).__init__()
        self.add_module('curried', func)
        self._curry_arguments = arguments
        self._curry_kwargs = keywords

    @property
    def name(self):
        return 'partial(%s, ...)' % self.curried.name

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
        return 'map_iid(%s, ...)' % self.func.name

    def iterate(self, trace, **kwargs):
        for item in self.map_items:
            kwargs = {**self.map_kwargs, **kwargs, 'trace': trace}
            result, trace = self.map_func(item, **kwargs)
            yield result

    def _forward(self, *args, **kwargs):
        return self.iterate(kwargs.pop('trace', None), **kwargs)

class Reduce(ModelSampler):
    def __init__(self, func, generator, initializer=None, **kwargs):
        super(Reduce, self).__init__()
        assert isinstance(func, Sampler)
        self.add_module('associative', func)
        if initializer is not None:
            assert isinstance(initializer, Sampler)
            self.add_module('initializer', initializer)
        else:
            self.initializer = None
        self._generator = generator
        self._associative_kwargs = kwargs

    @property
    def name(self):
        return 'reduce(%s, ...)' % self.associative.name

    def _forward(self, *args, **kwargs):
        if self.initializer is not None:
            accumulator, kwargs['trace'] = self.initializer(**kwargs)
        else:
            accumulator = None
        items = self._generator()
        for item in items:
            accumulator, kwargs['trace'] = self.associative(
                accumulator, item,
                *args, **kwargs, **self._associative_kwargs
            )
        return accumulator

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
    def name(self):
        formats = (self.sampler.name, self.particle_shape, self.before)
        return 'Population(%s, %s, before=%s)' % formats

    def _expand_args(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = utils.batch_expand(arg, self.particle_shape)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = utils.batch_expand(v, self.particle_shape)
        return tuple(args), kwargs

    def forward(self, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        return super(Population, self).forward(*args, **kwargs)

    def infer(self, results, trace):
        if not self.before:
            results = self._expand_args(*results)
        return results, trace

class HyperPopulation(PrimitiveCall):
    def __init__(self, primitive, particle_shape, name=None, trainable={},
                 hyper={}):
        super(HyperPopulation, self).__init__(primitive, name, trainable, hyper)
        self._particle_shape = particle_shape

    @property
    def particle_shape(self):
        return self._particle_shape

    @property
    def name(self):
        formats = (super(HyperPopulation, self).name, self.particle_shape)
        return 'HyperPopulation(%s, %s)' % formats

    def args_vardict(self):
        original = super(HyperPopulation, self).args_vardict()
        expander = lambda v: utils.batch_expand(v, self.particle_shape)
        return utils.vardict_map(original, expander)
