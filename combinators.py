#!/usr/bin/env python3

import collections
import inspect

import probtorch
import torch
import torch.nn as nn

import utils

class GraphingTrace(probtorch.stochastic.Trace):
    def __init__(self):
        super(GraphingTrace, self).__init__()
        self._modules = collections.defaultdict(lambda: {})
        self._stack = []

    def variable(self, Dist, *args, **kwargs):
        result = super(GraphingTrace, self).variable(Dist, *args, **kwargs)
        if self._stack:
            module_name = self._stack[-1]._function.__name__
            self._modules[module_name][kwargs['name']] = {
                'variable': self[kwargs['name']]
            }
        return result

    def push(self, module):
        self._stack.append(module)

    def pop(self):
        self._stack.pop()

    def annotation(self, module, variable):
        return self._modules[module][variable]

class Model(nn.Module):
    def __init__(self, f, phi={}, theta={}):
        super(Model, self).__init__()
        self._function = f
        self._trace = probtorch.Trace()
        self._observations = utils.EMPTY_TRACE.copy()
        self._parent = collections.defaultdict(lambda: None)
        self.condition()
        self.register_args(phi, True)
        self.register_args(theta, False)

    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, **kwargs):
            this = kwargs['this']
            temp = inner(*args, **kwargs)
            return outer(*temp, this=this) if isinstance(temp, tuple) else\
                   outer(temp, this=this)
        return result

    @classmethod
    def compose(cls, outer, inner, name=None):
        result = cls._bind(outer, inner)
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
        if isinstance(func, cls):
            result.add_module(func._function.__name__, func)
        for arg in arguments:
            if isinstance(arg, cls):
                result.add_module(arg._function.__name__, arg)
        for k, v in keywords.items():
            if isinstance(v, cls):
                result.add_module(k, v)
        return result

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

    def _condition(self, trace, observations):
        if trace is not None:
            self._trace = trace
        if observations is not None:
            self._observations = observations

    def condition(self, trace=None, observations=None):
        self.apply(lambda m: m._condition(trace, observations))

    @property
    def trace(self):
        return self._trace

    @property
    def observations(self):
        return self._observations

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
        if isinstance(self.trace, GraphingTrace):
            self.trace.push(self)
        result = self._function(*args, **kwargs)
        if isinstance(self.trace, GraphingTrace):
            self.trace.pop()
        return result

def sequence(step, T, *args, **kwargs):
    for t in range(T):
        args = step(*args, t, **kwargs)
    return args
