import inspect

import probtorch
import torch
import torch.nn as nn

import utils

class Model(nn.Module):
    def __init__(self, f, params_namespace='', phi={}, theta={}):
        super(Model, self).__init__()
        self._function = f
        self._params_namespace = params_namespace
        self.register_args(phi, True)
        self.register_args(theta, False)

    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, trace=probtorch.Trace(), **kwargs):
            temp = inner(*args, trace=trace, **kwargs)
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

    def forward(self, *args, trace=probtorch.Trace(), **kwargs):
        kwparams = {**self.kwargs_dict(), **kwargs}
        if self._params_namespace is not None:
            kwparams[self._params_namespace] = self.args_vardict(keep_vars=True)
        return self._function(*args, trace=trace, **kwparams)

class Conditionable(Model):
    def __init__(self, f, params_namespace='params'):
        super(Conditionable, self).__init__(f, params_namespace)

    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, trace=probtorch.Trace(), conditions=probtorch.Trace(),
                   **kwargs):
            temp = inner(*args, trace=trace, conditions=conditions, **kwargs)
            return outer(*temp, trace=trace, conditions=conditions)\
                   if isinstance(temp, tuple)\
                   else outer(temp, trace=trace, conditions=conditions)
        return result

    def forward(self, *args, trace=probtorch.Trace(),
                conditions=probtorch.Trace(), **kwargs):
        return self._function(*args, trace=trace, conditions=conditions,
                              **kwargs)

class Inference(Conditionable):
    @classmethod
    def _bind(cls, outer, inner):
        def result(*args, trace=probtorch.Trace(), conditions=probtorch.Trace(),
                   **kwargs):
            temp, trace = inner(*args, trace=trace, conditions=conditions,
                                **kwargs)
            return outer(*temp, trace=trace, conditions=conditions)\
                   if isinstance(temp, tuple)\
                   else outer(temp, trace=trace, conditions=conditions)
        return result
