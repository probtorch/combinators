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
        self._modules = collections.defaultdict(lambda: set())
        self._stack = []

    def variable(self, Dist, *args, **kwargs):
        result = super(GraphingTrace, self).variable(Dist, *args, **kwargs)
        if self._stack:
            self._modules[self._stack[-1]].add(self[kwargs['name']])
        return result

    def push(self, module):
        self._stack.append(module)

    def pop(self):
        self._stack.pop()

class Model(nn.Module):
    def __init__(self, f, params_namespace=None, phi={}, theta={}):
        super(Model, self).__init__()
        self._function = f
        self._params_namespace = params_namespace
        self.register_args(phi, True)
        self.register_args(theta, False)

    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, **kwargs):
            trace = kwargs['trace']
            temp = inner(*args, **kwargs)
            return outer(*temp, trace=trace) if isinstance(temp, tuple) else\
                   outer(temp, trace=trace)
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
        def result(*args, **kwargs):
            kwargs = {**kwargs, **keywords}
            return func(*arguments, *args, **kwargs)
        return cls(result)

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            if self._params_namespace is not None:
                k = self._params_namespace + '__' + k
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def args_vardict(self, keep_vars=False):
        result = self.state_dict(keep_vars=keep_vars)
        if self._params_namespace is not None:
            prefix = self._params_namespace + '__'
            result = {k[len(prefix):]: v for (k, v) in result.items()
                      if k.startswith(prefix)}
        return utils.vardict(result)

    def kwargs_dict(self):
        members = dict(self.__dict__, **self.state_dict(keep_vars=True))
        return {k: v for k, v in members.items()
                if k in inspect.signature(self._function).parameters.keys()}

    def forward(self, *args, trace={}, **kwargs):
        kwparams = {**self.kwargs_dict(), 'trace': trace, **kwargs}
        if self._params_namespace is not None:
            kwparams[self._params_namespace] = self.args_vardict(keep_vars=True)
        if isinstance(trace, GraphingTrace):
            trace.push(self)
        result = self._function(*args, **kwparams)
        if isinstance(trace, GraphingTrace):
            trace.pop()
        return result

class Conditionable(Model):
    def __init__(self, f, params_namespace=None):
        super(Conditionable, self).__init__(f, params_namespace)

    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, **kwargs):
            trace = kwargs['trace']
            conditions = kwargs['conditions']
            temp = inner(*args, **kwargs)
            return outer(*temp, trace=trace, conditions=conditions)\
                   if isinstance(temp, tuple)\
                   else outer(temp, trace=trace, conditions=conditions)
        return result

    def forward(self, *args, **kwargs):
        return super(Conditionable, self).forward(*args, **kwargs)

class Inference(Conditionable):
    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, **kwargs):
            temp = inner(*args, **kwargs)
            outer_kwargs = {'trace': temp[-1],
                            'conditions': kwargs['conditions']}
            return outer(*temp[:-1], **outer_kwargs)
        return result

    def forward(self, *args, **kwargs):
        trace = kwargs['trace']
        result = super(Inference, self).forward(*args, **kwargs)
        return result if isinstance(result, tuple) else (result, trace)

def sequence(step, T, *args, trace={}, conditions={}):
    for t in range(T):
        results = step(*args, t, trace=trace, conditions=conditions)
        if isinstance(results, tuple):
            args = results[:-1]
            trace = results[-1]
        else:
            args = results
    return results
