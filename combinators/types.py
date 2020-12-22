""" type aliases """

from torch import Tensor
from probtorch import Trace, Factor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable

State = Any
Samples = Union[Any, List[Any]]
TraceLike = Union[Trace, Dict[str, Tensor]]
