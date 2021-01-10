""" type aliases """

from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from combinators.stochastic import Trace, Factor, GenericRandomVariable
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

def get_shape_kwargs(fn, sample_dims=None, batch_dim=None):
    kwargs = dict()
    if check_passable_kwarg('sample_dims', fn):
        kwargs['sample_dims'] = sample_dims
    if check_passable_kwarg('batch_dim', fn):
        kwargs['batch_dim'] = batch_dim
    return kwargs
