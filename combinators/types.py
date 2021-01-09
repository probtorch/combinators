""" type aliases """

from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from combinators.stochastic import Trace, Factor, GenericRandomVariable

State = Any
Output = Union[Any, List[Any]]
TraceLike = Union[Trace, Dict[str, Union[Tensor, GenericRandomVariable]]]

def get_value(t:TraceLike, k:str):
    val = t[k]
    return val if isinstance(val, Tensor) else val.value
