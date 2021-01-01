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

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program, model
from combinators.traceable import Traceable


class Kernel(Traceable, nn.Module):
    """ superclass of a program? """
    def __init__(self) -> None:
        super().__init__()

    # TODO: do we need *args? I am thinking no for the time being
    @abstractmethod
    def apply_kernel(self, trace: Trace, cond_trace: Trace, outs: Output) -> Output:
        raise NotImplementedError()

    def forward(self, cond_trace: Trace, outs: Output) -> Tuple[Trace, Output]:
        # get a fresh trace
        trace = self.get_trace(evict=True)

        # apply the kernel
        out = self.apply_kernel(trace, cond_trace, outs)

        # FIXME: momentary hack
        for name in cond_trace.keys():
            trace.append(cond_trace[name], name=f'{name}_in')


        # TODO: update the trace? is this important?
        # trace.update(copytrace(cond_trace, set(cond_trace.keys() - set(trace.keys()))))


        return trace, out

