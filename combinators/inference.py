#!/usr/bin/env python3
#
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

import combinators.utils as trace_utils
from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program
from combinators.kernel import Kernel

class Inf:
    pass

class KernelInf:
    pass

@typechecked
class Reverse(nn.Module, KernelInf, Inf):
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
class Forward(nn.Module, KernelInf, Inf):
    """ FIXME: Reverse and Forward seem wrong """
    def __init__(self, kernel: Kernel, target: Program) -> None:
        super().__init__()
        self.target = target
        self.kernel = kernel

    def forward(self, *program_args:Any) -> Tuple[Trace, Output]:
        tr, out = self.target(*program_args)
        return self.kernel(tr, out)

@typechecked
class Propose(nn.Module, Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Program, Inf], batch_dim=None):
        super().__init__()
        self.target = target
        self.proposal = proposal
        self.batch_dim = batch_dim

    def forward(self, *target_args):
        target_trace, _ = self.target(*target_args)

        # FIXME: make sure pytorch post-forward hooks are run at the correct time.
        # def run_proposal(*proposal_args):
        #     return self.propsal(*proposal_args, trace=target_trace)
        # return run_proposal
        return self.proposal(*target_args, cond_trace=target_trace)


    @property
    def log_weights(self):
        target_trace = self.target.get_trace()
        proposal_trace = self.proposal.get_trace()
        return proposal_trace.log_joint(batch_dim=self.batch_dim) - target_trace.log_joint(batch_dim=self.batch_dim)

    @property
    def trace(self):
        ttr = self.target.get_trace()
        ptr = self.proposal.get_trace()
        return Trace(list(d1.items()) + list(d2.items()))
