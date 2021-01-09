#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
import combinators.trace.utils as trace_utils

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program, model
from combinators.traceable import TraceModule


class Kernel(TraceModule):
    """ superclass of a program? """
    def __init__(self) -> None:
        super().__init__()

    # TODO: do we need *args? I am thinking no for the time being
    @abstractmethod
    def apply_kernel(self, trace: Trace, cond_trace: Trace, outs: Output) -> Output:
        raise NotImplementedError()

    def forward(self, cond_trace: Trace, cond_outs: Output) -> Tuple[Trace, Output]:
        # get a fresh trace to make sure we don't have inplace mutation
        trace = Trace()

        out = self.apply_kernel(trace, cond_trace, cond_outs)

        # grab anything that is missing from the cond_trace
        full_trace = trace_utils.copytraces(cond_trace, trace)

        return full_trace, out
