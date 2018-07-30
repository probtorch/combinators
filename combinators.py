#!/usr/bin/env python3

import collections
import functools
import inspect

import probtorch
from probtorch.stochastic import RandomVariable
import torch
import torch.nn as nn

import utils

class ParticleTrace(probtorch.stochastic.Trace):
    def __init__(self, num_particles=1):
        super(ParticleTrace, self).__init__()
        self._modules = collections.defaultdict(lambda: {})
        self._stack = []
        self._num_particles = num_particles

    @property
    def num_particles(self):
        return self._num_particles

    def log_joint(self, *args, **kwargs):
        return super(ParticleTrace, self).log_joint(*args, sample_dim=0,
                                                    **kwargs)

    def variable(self, Dist, *args, **kwargs):
        args = [arg.expand(self.num_particles, *arg.shape)
                if isinstance(arg, torch.Tensor) and
                (len(arg.shape) < 1 or arg.shape[0] != self.num_particles)
                else arg for arg in args]
        kwargs = {k: v.expand(self.num_particles, *v.shape)
                     if isinstance(v, torch.Tensor) and
                     (len(v.shape) < 1 or v.shape[0] != self.num_particles)
                     else v for k, v in kwargs.items()}
        result = super(ParticleTrace, self).variable(Dist, *args, **kwargs)
        if self._stack:
            module_name = self._stack[-1]._function.__name__
            self._modules[module_name][kwargs['name']] = {
                'variable': self[kwargs['name']]
            }
        return result

    def squeeze(self):
        result = self.__class__()
        result._modules = self._modules
        result._stack = self._stack

        for i, key in enumerate(self.variables()):
            if key is not None:
                rv = self[key]
                result[key] = RandomVariable(rv.dist, rv.value.median(dim=0)[0],
                                             rv.observed, rv.mask,
                                             rv.reparameterized)
            else:
                rv = self[i]
                result[i] = RandomVariable(rv.dist, rv.value.median(dim=0)[0],
                                           rv.observed, rv.mask,
                                           rv.reparameterized)

        return result

    def push(self, module):
        self._stack.append(module)

    def pop(self):
        self._stack.pop()

    def annotation(self, module, variable):
        return self._modules[module][variable]

    def has_annotation(self, module, variable):
        return variable in self._modules[module]

    def keys(self):
        return self._nodes.keys()

class Model(nn.Module):
    def __init__(self, f, phi={}, theta={}):
        super(Model, self).__init__()
        if isinstance(f, Model):
            self.add_module('_function', f)
        else:
            self._function = f
        self._trace = None
        self._guide = None
        self._parent = collections.defaultdict(lambda: None)
        self.register_args(phi, True)
        self.register_args(theta, False)

    @classmethod
    def _bind(cls, outer, inner, intermediate_name=None):
        def result(*args, **kwargs):
            this = kwargs['this']
            temp = inner(*args, **kwargs)
            if intermediate_name:
                kws = {'this': this, intermediate_name: temp}
                return outer(**kws)
            return outer(*temp, this=this) if isinstance(temp, tuple) else\
                   outer(temp, this=this)
        return result

    @classmethod
    def compose(cls, outer, inner, name=None, intermediate_name=None):
        result = cls._bind(outer, inner, intermediate_name)
        if name is not None:
            result.__name__ = name
        result = cls(result)

        for (i, element) in enumerate([inner, outer]):
            if isinstance(element, nn.Module):
                elt_name = element._function.__name__\
                           if isinstance(element, cls) else 'element%d' % i
                result.add_module(elt_name, element)

        return result

    @classmethod
    def partial(cls, func, *arguments, **keywords):
        def wrapper(*args, **kwargs):
            kwargs = {**kwargs, **keywords}
            return func(*arguments, *args, **kwargs)
        result = cls(wrapper)
        if isinstance(func, Model):
            result.add_module(func.name, func)
        for arg in arguments:
            if isinstance(arg, cls):
                result.add_module(arg.name, arg)
        for k, v in keywords.items():
            if isinstance(v, cls):
                result.add_module(k, v)
        return result

    @classmethod
    def map_iid(cls, func, items, **kwargs):
        return cls(lambda **kws: [func(item, **kws, **kwargs) for item in items])

    @classmethod
    def reduce(cls, func, items, initializer=None, **kwargs):
        def wrapper(*args, items=items, initializer=initializer, **kws):
            return functools.reduce(
                functools.partial(func, *args, **kws, **kwargs), items,
                initializer
            )
        result = cls(wrapper)
        if isinstance(func, Model):
            result.add_module(func.name, func)
        return result

    @classmethod
    def sequence(cls, step, T, *args, **kwargs):
        return cls.reduce(step, range(T), initializer=args, **kwargs)

    @property
    def name(self):
        return self._function.__name__

    @property
    def __name__(self):
        return self.name

    def add_module(self, name, module):
        super(Model, self).add_module(name, module)
        if isinstance(module, Model):
            module._parent['parent'] = self

    @property
    def parent(self):
        return self._parent['parent']

    @property
    def ancestor(self):
        result = self.parent
        while result.parent is not None:
            result = result.parent
        return result

    def _condition(self, trace, guide):
        if trace is not None:
            self._trace = trace
        if guide is not None:
            self._guide = guide

    def _condition_all(self, trace=None, guide=None):
        if trace is None:
            trace = ParticleTrace()
        if guide is None:
            guide = utils.EMPTY_TRACE.copy()
        self.apply(lambda m: m._condition(trace, guide))

    @property
    def trace(self):
        return self._trace

    @property
    def guide(self):
        return self._guide

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def args_vardict(self, keep_vars=True):
        return utils.vardict(self.state_dict(keep_vars=keep_vars))

    def forward(self, *args, **kwargs):
        kwargs = {**kwargs, 'this': self}
        if not self.parent:
            self._condition_all(trace=kwargs.pop('trace', None),
                                guide=kwargs.pop('guide', None))
        if isinstance(self.trace, ParticleTrace):
            self.trace.push(self)
        result = self._function(*args, **kwargs)
        if isinstance(self.trace, ParticleTrace):
            self.trace.pop()
        return result

    def simulate(self, *args, **kwargs):
        if 'trace' in kwargs:
            trace = kwargs.pop('trace')
        else:
            trace = ParticleTrace()
        guide = kwargs.pop('guide') if 'guide' in kwargs else None
        self.condition(trace, guide)
        result = self.forward(*args, **kwargs)
        return result, self.trace.log_joint()
