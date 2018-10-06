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

import utils

class BroadcastingTrace(probtorch.stochastic.Trace):
    def __init__(self, num_particles=1):
        super(BroadcastingTrace, self).__init__()
        self._modules = collections.defaultdict(lambda: {})
        self._stack = []
        self._particle_stack = [num_particles]

    @property
    def batch_shape(self):
        return tuple(self._particle_stack)

    def push_batch(self, n):
        self._particle_stack = [n] + self._particle_stack

    def pop_batch(self):
        result = self._particle_stack[1:]
        self._particle_stack = result
        return result

    @property
    def num_particles(self, i=-1):
        if i >= 0:
            return self._particle_stack[i]
        return int(np.product(self._particle_stack))

    def log_joint(self, *args, **kwargs):
        return super(BroadcastingTrace, self).log_joint(*args, sample_dim=0,
                                                        **kwargs)

    def variable(self, Dist, *args, **kwargs):
        to_shape = kwargs.pop('to', None)
        args = [utils.batch_expand(arg, to_shape)
                if isinstance(arg, torch.Tensor) and to_shape
                else arg for arg in args]
        kwargs = {k: utils.batch_expand(v, to_shape)
                     if isinstance(v, torch.Tensor) and to_shape else v
                  for k, v in kwargs.items()}
        result = super(BroadcastingTrace, self).variable(Dist, *args, **kwargs)
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

    def unwrap(self, predicate=lambda k, rv: True):
        result = collections.OrderedDict()

        for i, key in enumerate(self.variables()):
            if key is not None:
                if predicate(key, self[key]):
                    result[key] = self[key].value.median(dim=0)[0]
            elif predicate(i, self[i]):
                result[i] = self[i].value.median(dim=0)[0]

        return result

    def push(self, module):
        self._stack.append(module)

    def pop(self):
        self._stack.pop()

    def annotation(self, module, variable):
        return self._modules[module][variable]

    def has_annotation(self, module, variable):
        return variable in self._modules[module]

    def have_annotation(self, modules, variable):
        return any([variable in self._modules[module] for module in modules])

    def keys(self):
        return self._nodes.keys()

    def is_observed(self, name):
        return self.observed(name) is not None

    def observed(self, name):
        if name in self._nodes and self._nodes[name].observed:
            return self._nodes[name]
        return None

    def clamped(self, name):
        return self.observed(name)

    @property
    def observations(self):
        return [rv for rv in self.variables() if self.is_observed(rv) and\
                self.have_annotation(self._modules.keys(), rv)]

    @property
    def latents(self):
        return [rv for rv in self.variables()\
                if not self.is_observed(rv) and\
                self.have_annotation(self._modules.keys(), rv)]

class ConditionedTrace(BroadcastingTrace):
    def __init__(self, num_particles=1, guide=None, data=None):
        super(ConditionedTrace, self).__init__(num_particles)
        self._guide = guide
        self._data = data

    @classmethod
    def clamp(cls, guide):
        return guide.__class__(guide.num_particles, guide=guide,
                               data=guide.data)

    @property
    def guide(self):
        return self._guide

    @property
    def data(self):
        return self._data

    def observed(self, name):
        if self._data and name in self._data:
            return self._data[name]
        return None

    def guided(self, name):
        if self._guide and name in self._guide:
            return self._guide[name].value
        return None

    def clamped(self, name):
        observed = self.observed(name)
        guided = self.guided(name)
        if observed is not None:
            return observed
        elif guided is not None:
            return guided
        return None

    def variable(self, Dist, *args, **kwargs):
        if 'name' in kwargs and not kwargs.get('value', None):
            name = kwargs['name']
            clamped = self.clamped(name)
            to_shape = kwargs.get('to', None)
            if isinstance(clamped, torch.Tensor) and not to_shape and\
               self.observed(name) is not None:
                clamped = utils.batch_expand(clamped, self.batch_shape)
            if clamped is not None:
                kwargs['value'] = clamped

        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if tensors and 'value' in kwargs and kwargs['value'] is not None:
            kwargs['value'] = kwargs['value'].to(device=tensors[0].device)
        return super(ConditionedTrace, self).variable(Dist, *args, **kwargs)

    def log_joint(self, *args, normalize_guide=False, **kwargs):
        generative_joint = super(ConditionedTrace, self).log_joint(*args, **kwargs)
        if isinstance(generative_joint, torch.Tensor):
            device = generative_joint.device
        elif len(self):
            device = self[list(self.variables())[0]].value.device
        else:
            device = 'cpu'
        if normalize_guide and self.guide:
            all_nodes = kwargs.get('nodes', None) or self.keys()
            guided_nodes = [node for node in all_nodes
                            if self.guided(node) is not None and
                            self.observed(node) is None]
            guide_joint = self._guide.log_joint(
                *args, nodes=guided_nodes,
                reparameterized=kwargs.get('reparameterized', True)
            )
        else:
            guide_joint = torch.zeros(self.batch_shape).to(device)

        return generative_joint - guide_joint

class Model(nn.Module):
    def __init__(self, f, trainable={}, hyper={}):
        super(Model, self).__init__()
        if isinstance(f, Model):
            self.add_module('_function', f)
        else:
            self._function = f
        self._trace = None
        self._parent = collections.defaultdict(lambda: None)
        self.register_args(trainable, True)
        self.register_args(hyper, False)

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

    def _condition(self, trace):
        if trace is not None:
            self._trace = trace

    def _condition_all(self, trace=None):
        if trace is None:
            trace = BroadcastingTrace()
        self.apply(lambda m: m._condition(trace))

    @property
    def trace(self):
        return self._trace

    @property
    def guided(self):
        return isinstance(self._trace, ConditionedTrace)

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            v = torch.tensor(v)
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def args_vardict(self, to, keep_vars=True):
        return utils.vardict(self.state_dict(keep_vars=keep_vars), to=to)

    def forward(self, *args, **kwargs):
        kwargs = {**kwargs, 'this': self}
        if kwargs.pop('separate_traces', False) or not self.parent:
            self._condition_all(trace=kwargs.pop('trace', None))
        if isinstance(self.trace, BroadcastingTrace):
            self.trace.push(self)
        result = self._function(*args, **kwargs)
        if isinstance(self.trace, BroadcastingTrace):
            self.trace.pop()
        return result

    def simulate(self, *args, **kwargs):
        if 'trace' not in kwargs:
            kwargs['trace'] = BroadcastingTrace()
        reparameterized = kwargs.pop('reparameterized', True)

        result = self.forward(*args, **kwargs)
        return self.trace.log_joint(reparameterized=reparameterized), result
