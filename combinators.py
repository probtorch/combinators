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

    def args_vardict(self):
        result = utils.vardict(self.state_dict(keep_vars=True))
        # PyTorch BUG: Parameter's don't get counted as Tensors in Normal
        for k, v in result.items():
            result[k] = v.clone()
        return result

class ModelSampler(Sampler):
    def forward(self, *args, **kwargs):
        if 'trace' not in kwargs:
            kwargs['trace'] = trace_tries.HierarchicalTrace()
        return self._forward(*args, **kwargs)

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

    def _forward(self, *args, **kwargs):
        params = self.args_vardict()
        if len(params):
            kwargs['params'] = params
        trace = kwargs['trace']
        kwargs['trace'] = trace.extract(self.name)
        result = self.primitive(*args, **kwargs)
        trace.insert(self.name, kwargs.pop('trace'))
        return result, trace

class InferenceSampler(Sampler):
    def __init__(self, sampler):
        super(InferenceSampler, self).__init__()
        assert isinstance(sampler, Sampler)
        self.add_module('sampler', sampler)

    @property
    def name(self):
        return self.sampler.name

    def forward(self, *args, **kwargs):
        trace = kwargs.pop('trace')
        trace, args, kwargs = self.sample_prehook(trace, *args, **kwargs)
        kwargs['trace'] = trace
        return self.sample_hook(*self.sampler(*args, **kwargs))

    def sample_prehook(self, trace, *args, **kwargs):
        raise NotImplementedError()

    def sample_hook(self, results, trace):
        raise NotImplementedError()

class Score(InferenceSampler):
    def sample_prehook(self, trace, *args, **kwargs):
        trace = trace_tries.HierarchicalTrace(proposal=trace)
        return trace, args, kwargs

    def sample_hook(self, results, trace):
        return results, trace

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

# TODO: Ultimately, I want to decompose this somehow to help me implement Trace
# MCMC.  In specific, I want to be able to:
# * Run a program, extracting a trace
# * Run a program clamped to a trace, properly weighting the new trace (Score)
# * Generate new traces from old

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
        return 'Reduce(%s)' % self.associative.name

    def _forward(self, *args, **kwargs):
        trace = kwargs.pop('trace')
        if self.initializer is not None:
            kwargs['trace'] = trace.extract(self.initializer.name)
            accumulator, init_trace = self.initializer(**kwargs)
            trace.insert(self.initializer.name, init_trace)
        else:
            accumulator = None
        items = self._generator()
        for item in items:
            kwargs['trace'] = trace.extract(self.name + '/' + str(item))
            accumulator, step_trace = self.associative(
                accumulator, item,
                *args, **kwargs, **self._associative_kwargs
            )
            trace.insert(self.name + '/' + str(item), step_trace)
        return accumulator, trace

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

    def _expand_args(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = utils.batch_expand(arg, self.particle_shape)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = utils.batch_expand(v, self.particle_shape)
        return tuple(args), kwargs

    def sample_prehook(self, trace, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        return trace, args, kwargs

    def sample_hook(self, results, trace):
        if not self.before:
            results = self._expand_args(*results)
        return results, trace

class HyperPopulation(PrimitiveCall):
    def __init__(self, primitive, particle_shape, name=None, trainable={},
                 hyper={}):
        super(HyperPopulation, self).__init__(primitive, name, trainable,
                                              hyper)
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
