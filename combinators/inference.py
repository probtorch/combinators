#!/usr/bin/env python3
#
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
from typing import Iterable

import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Observable

State = namedtuple("State", ['trace', 'output'])

optional = lambda x, get: getattr(x, get) if x is not None else None

@typechecked
class State(Iterable):
    def __init__(self, trace:Optional[Trace], output:Optional[Output]):
        self.trace = trace
        self.output = output
    def __repr__(self):
        if self.trace is None:
            return "None"
        else:
            out_str = tensor_utils.show(self.output) if isinstance(self.output, Tensor) else self.output
            return "tr: {}; out: {}".format(trace_utils.show(self.trace), out_str)
    def __iter__(self):
        for x in [self.trace, self.output]:
            yield x

@typechecked
class KCache:
    def __init__(self, program:Optional[State], kernel:Optional[State]):
        self.program = program
        self.kernel = kernel
    def __repr__(self):
        return "Kernel Cache:" + \
            "\n  program: {}".format(self.program) + \
            "\n  kernel:  {}".format(self.kernel)

@typechecked
class PCache:
    def __init__(self, target:Optional[State], proposal:Optional[State]):
        self.target = target
        self.proposal = proposal
    def __repr__(self):
        return "Propose Cache:" + \
            "\n  proposal: {}".format(self.proposal) + \
            "\n  target:   {}".format(self.target)

class Inf:
    pass

class KernelInf(nn.Module, Observable):
    def __init__(self):
        nn.Module.__init__(self)
        Observable.__init__(self)
        self._cache = KCache(None, None)

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))


@typechecked
class Reverse(KernelInf, Inf):
    def __init__(self, program: Program, kernel: Kernel) -> None:
        super().__init__()
        self.program = program
        self.kernel = kernel

    def new_forward(self, cond_trace:Trace, *program_args:Any) -> Tuple[Trace, Output]:
        self.program.update_conditions(self.observations)
        program_state = State(*self.program(*program_args))
        self.program.clear_conditions()

        self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, *program_args))
        self.kernel.clear_conditions()

        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, None

    def forward(self, *program_args:Any) -> Tuple[Trace, Output]:
        self.program.update_conditions(self.observations)
        program_state = State(*self.program(*program_args))
        self.program.clear_conditions()

        self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, *program_args))
        self.kernel.clear_conditions()

        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, None


@typechecked
class Forward(KernelInf, Inf):
    def __init__(self, kernel: Kernel, program: Program) -> None:
        super().__init__()
        self.program = program
        self.kernel = kernel

    def new_forward(self, *program_args:Any) -> Tuple[Trace, Output]:
        program_state = State(*self.program(*program_args))

        self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, *program_args))
        self.kernel.clear_conditions()
        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, kernel_state.output

    def forward(self, *program_args:Any) -> Tuple[Trace, Output]:
        program_state = State(*self.program(*program_args))
        # # stop gradients for nesting. FIXME: is this the correct location?
        # for k, v in program_state.trace.items():
        #     program_state.trace[k]._value = program_state.trace[k].value.detach()

        self.kernel.update_conditions(self.observations)
        kernel_state = State(*self.kernel(*program_state, *program_args))
        self.kernel.clear_conditions()
        self._cache = KCache(program_state, kernel_state)
        return kernel_state.trace, kernel_state.output


@typechecked
class Propose(nn.Module, Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Program, Inf], validate:bool=True):
        super().__init__()
        self.target = target
        self.proposal = proposal
        self._cache = PCache(None, None)
        self.validate = validate

    def forward(self, *args):
        # FIXME: target and proposal args can / should be separated
        proposal_state = State(*self.proposal(*args))

        self.target.condition_on(proposal_state.trace)
        target_state = State(*self.target(*args))
        self.target.clear_conditions()
        proposal_state.trace['q']._value = proposal_state.trace['q'].value.detach()

        self._cache = PCache(target_state, proposal_state)
        return self._cache, Propose.log_weights(target_state.trace, proposal_state.trace, self.validate)

    @classmethod
    def log_weights(cls, target_trace, proposal_trace, validate=True):
        if validate:
            assert trace_utils.valeq(proposal_trace, target_trace, nodes=target_trace._nodes, check_exist=True)

        return target_trace.log_joint(batch_dim=None, sample_dims=0, nodes=target_trace._nodes) - \
            proposal_trace.log_joint(batch_dim=None, sample_dims=0, nodes=target_trace._nodes)

