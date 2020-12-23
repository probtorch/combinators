""" type aliases """

from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from combinators.stochastic import Trace, Factor

State = Any
Output = Union[Any, List[Any]]
TraceLike = Union[Trace, Dict[str, Tensor]]
