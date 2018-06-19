import probtorch
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, f, params_namespace='params'):
        super(Model, self).__init__()
        self._function = f

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

    def forward(self, *args, trace=probtorch.Trace(), **kwargs):
        return self._function(*args, trace=trace, **kwargs)

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
