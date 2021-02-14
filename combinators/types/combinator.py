#!/usr/bin/env python3

from probtorch import Trace
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typeguard import typechecked
from typing import Tuple, NewType, Any


Output = Any

Weight = NewType("Weight", Tensor)


class Combinator(metaclass=ABCMeta):
    """
    A superclass for inference combinators (for maybe involving model combinators)
    """
    @abstractmethod
    @typechecked
    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...
