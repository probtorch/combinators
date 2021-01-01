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
from combinators.utils import assert_valid_subtrace, copytrace
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

        # assert that the trace computed the right thing for importance weights
        assert_valid_subtrace(cond_trace, trace)
        # FIXME: momentary hack
        for name in cond_trace.keys():
            trace.append(cond_trace[name], name=f'{name}_in')


        # TODO: update the trace? is this important?
        # trace.update(copytrace(cond_trace, set(cond_trace.keys() - set(trace.keys()))))


        return trace, out


@typechecked
class Reverse(nn.Module):
    """ FIXME: Reverse and Forward seem wrong """
    def __init__(self, proposal: Program, kernel: Kernel) -> None:
        super().__init__()
        self.proposal = proposal
        self.kernel = kernel

    def forward(self, *program_args:Any) -> Trace:
        tr, out = self.proposal(*program_args)
        ktr, _ = self.kernel(tr, out)
        return ktr

@typechecked
class Forward(nn.Module):
    """ FIXME: Reverse and Forward seem wrong """
    def __init__(self, kernel: Kernel, target: Program) -> None:
        super().__init__()
        self.target = target
        self.kernel = kernel

    def forward(self, *program_args:Any) -> Tuple[Trace, Output]:
        tr, out = self.target(*program_args)
        return self.kernel(tr, out)
