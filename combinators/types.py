""" type aliases """

from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from combinators.stochastic import Trace, Factor, GenericRandomVariable
import combinators.tensor.utils as tensor_utils
import inspect

State = Any
Output = Union[Any, List[Any]]
TraceLike = Union[Trace, Dict[str, Union[Tensor, GenericRandomVariable]]]

def get_value(t:TraceLike, k:str):
    val = t[k]
    return val if isinstance(val, Tensor) else val.value

def check_passable_kwarg(name, fn):
    fullspec = inspect.getfullargspec(fn)
    return fullspec.varkw is not None or name in fullspec.kwonlyargs or name in fullspec.args

def check_passable_arg(name, fn):
    fullspec = inspect.getfullargspec(fn)
    return fullspec.varargs is not None or name in fullspec.args

def get_shape_kwargs(fn, sample_dims=None, batch_dim=None):
    kwargs = dict()
    if check_passable_kwarg('sample_dims', fn):
        kwargs['sample_dims'] = sample_dims
    if check_passable_kwarg('batch_dim', fn):
        kwargs['batch_dim'] = batch_dim
    return kwargs


class property_dict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        return "<property_dict>:\n" + show_nest(self, pd_header="<property_dict>")

def show_nest(p:property_dict, nest_level=0, indent_len:Optional[int]=None, pd_header="<property_dict>"):
    _max_len = max(map(len, p.keys()))
    max_len = _max_len + nest_level * (_max_len if indent_len is None else indent_len)
    delimiter = "\n  "

    unnested = dict(filter(lambda kv: not isinstance(kv[1], property_dict), p.items()))
    unnested_str = delimiter.join([
        *[("{:>"+ str(max_len)+ "}: {}").format(k, tensor_utils.show(v) if isinstance(v, Tensor) else v) for k, v in unnested.items()
         ]
    ])

    nested = dict(filter(lambda kv: isinstance(kv[1], property_dict), p.items()))
    nested_str = delimiter.join([
        *[("{:>"+ str(max_len)+ "}: {}").format(k + pd_header, "\n"+show_nest(v, nest_level=nest_level+1)) for k, v in nested.items()
         ]
    ])

    return unnested_str + delimiter + nested_str

PropertyDict = property_dict

class iproperty_dict(property_dict):
    def __iter__(self):
        for v in self.values():
            yield v

IPropertyDict = iproperty_dict

class Out(PropertyDict):
    def __init__(self, trace:Trace, log_joint:Optional[Output], output:Output, extras:dict=dict()):
        self.trace = trace
        self.log_joint = log_joint
        self.output = output
        for k, v in extras.items():
            self[k] = v

    def __iter__(self):
        for x in [self.trace, self.log_joint, self.output]:
            yield x
