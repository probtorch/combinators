#!/usr/bin/env python3

import torch
import torch.nn as nn
from probtorch import Trace, Factor
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod

from combinators.program import Program, model
from combinators.utils import assert_valid_subtrace, copytrace
from combinators.types import Output, State, TraceLike


@typechecked
class Kernel:  #(Program): maybe?
    def __init__(self, program:Program):
        self.trace = Trace()
        self.program = program

    @abstractmethod
    def evaluate(self, kernel_trace:Trace, cond_trace:Trace, obs:Output) -> Tuple[Trace, Output]:
        """ evaluate observations under given trace, return logprobs """
        raise NotImplementedError()

    def _evaluate(self, cond_trace:Trace, obs:Output) -> Tuple[Trace, Output]:
        """ evaluate the samples under the conditioned trace, returning the trace """
        trace_out, samples_out = self.evaluate(self.trace, cond_trace, obs)
        # FIXME: this needs to be pure!!!
        trace_out.update(copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys()))))
        return trace_out, samples_out

    # Sort of what I thought would be the entire thing:
    # def _generate(self, state: State, cond_trace: Trace) -> Tuple[State, Trace, Output]:
    #     """ produce samples for a RV that I produce, returning the samples and trace """
    #     state_out,       out = self.generate(state, cond_trace)
    #     state_out, trace_out = self.evaluate(state, cond_trace, out)
    #     trace_out = trace_out + copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys())))
    #     return state_out, trace_out, out

    def __call__(self, *args, **kwargs) -> Callable[[], Tuple[Trace, Output]]:
        prg_trace, prg_samples = self.program(*args, **kwargs)

        def run_kernel():
            trace_out, samples_out = self._evaluate(prg_trace, prg_samples)
            # tr_{prg} <: tr_{ker} -- ie: all of kernel tr in prg tr
            assert_valid_subtrace(trace_out, self.trace)
            return trace_out, samples_out

        return run_kernel


class Reverse:
    def __init__(self, proposal: Program, kernel: Kernel):
        self.proposal = proposal
        self.kernel = kernel

    def __call__(self, state: State) -> Tuple[State, Trace, Output]:
        return self.kernel._generate(state, self.proposal.trace)


class Forward:
    def __init__(self, kernel: Kernel, target: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Output) -> Tuple[State, Trace, Output]:
        return self.kernel._evaluate(state, self.proposal.trace, x)


class Propose:
    def __init__(self, target: Program, proposal: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Output) -> Tuple[State, Trace, Output]:
        return self.kernel._evaluate(state, self.proposal.trace, x)
