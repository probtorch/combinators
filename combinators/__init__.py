#!/usr/bin/env python3

import torch
import torch.nn as nn
from probtorch import Trace, Factor
from torch import Tensor
from typing import Callable, Any, Tuple, Optional
from collections import ChainMap

from combinators.utils import assert_valid_subtrace, copytrace

State   = Any
Samples = Tensor


class Program:
    """ superclass of a program? """
    def __init__(self):
        self.trace = Trace()

    # def log_probs(self, state: State, inputs: Tensor) -> Tuple[State, Trace]:
    #     """ TODO: make this implicit """
    #     raise NotImplementedError()

    def evaluate(self, trace: Trace, inputs: Tensor) -> Tuple[Trace, Samples]:
        raise NotImplementedError()

    def _evaluate(self, state: State, trace: Trace, inputs: Tensor) -> Tuple[Trace, Samples]:
        raise NotImplementedError()

    def __call__(self, state: State, inputs: Tensor, trace: Optional[Trace] = None) -> Tuple[State, Trace, Samples]:
        state_out, trace_out = self.log_probs(state, inputs) if trace is None else trace
        state_out, out = self.evaluate(state_out, trace_out, inputs)
        # TODO: enforce purity?
        return state_out, trace_out, out

class P(Program):
    def __init__(self):
        super().__init__()
        self.p1 = P1(...)  # <<< definitely need to think about this

    def evaluate(self, state: State, trace: Trace, inputs: Tensor) -> Tuple[Trace, Samples]:
        self.p1(inputs)


class Kernel:
    def __init__(self):
        self.trace = Trace()

    def evaluate(self, state: State, cond_trace: Trace, obs: Samples) -> Tuple[State, Trace]:
        """ evaluate observations under given trace, return logprobs """
        # automatically implied under the hood
        raise NotImplementedError()

    def _evaluate(self, state: State, cond_trace: Trace, obs: Samples) -> Tuple[State, Trace, Samples]:
        """ evaluate the samples under the conditioned trace, returning the trace """
        state_out, trace_out = self.evaluate(state, cond_trace, obs)
        trace_out = trace_out + copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys())))
        return state_out, trace_out, obs

    def generate(self, state: State, cond_trace: Trace) -> Tuple[State, Samples]:
        """ generate logprobs, run them through kernel with program tr """
        raise NotImplementedError()

    def _generate(self, state: State, cond_trace: Trace) -> Tuple[State, Trace, Samples]:
        """ produce samples for a RV that I produce, returning the samples and trace """
        state_out,       out = self.generate(state, cond_trace)
        state_out, trace_out = self.evaluate(state, cond_trace, out)
        trace_out = trace_out + copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys())))
        return state_out, trace_out, out

    def __call__(self, state: State, cond_trace: Trace, x: Samples = None) -> Tuple[State, Trace, Samples]:
        # tr_{prg} <: tr_{ker} -- ie: all of kernel tr in prg tr
        assert_valid_subtrace(cond_trace, self.trace)

        if x is None:
            return self._generate(state, cond_trace)
        else:
            return self._evaluate(state, cond_trace, x)


class Reverse:
    def __init__(self, proposal: Program, kernel: Kernel):
        self.proposal = proposal
        self.kernel = kernel

    def __call__(self, state: State) -> Tuple[State, Trace, Samples]:
        return self.kernel._generate(state, self.proposal.trace)


class Forward:
    def __init__(self, kernel: Kernel, target: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Samples) -> Tuple[State, Trace, Samples]:
        return self.kernel._evaluate(state, self.proposal.trace, x)


class Propose:
    def __init__(self, target: Program, proposal: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Samples) -> Tuple[State, Trace, Samples]:
        return self.kernel._evaluate(state, self.proposal.trace, x)
