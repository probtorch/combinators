#!/usr/bin/env python3
from probtorch import Trace
from torch import Tensor
from typing import NewType, Callable, Protocol

from combinators.types.combinator import Output, Weight
from combinators.types.model import Program


class Kernel(object):
    """
    A kernel runs an operation conditioned on a program.

    """

    def __init__(self, program: Program):
        self.program = program

    def __call__(self, samples: Tensor) -> Output:
        raise NotImplementedError()

    def evaluate(self, samples: Tensor, trace: Trace) -> Tensor:
        raise NotImplementedError()

    def weight(self, samples: Tensor, trace: Trace) -> Weight:
        raise NotImplementedError()

    def dist(self, samples: Tensor, trace: Trace) -> Tensor:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "Kernel({})".format(repr(self.program))
