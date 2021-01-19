#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor, distributions
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
from typing import Iterable

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs
from combinators.trace.utils import RequiresGrad, copytrace
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Conditionable
from inspect import signature
import inspect
from combinators.objectives import nvo_avo

@typechecked
class State(Iterable):
    def __init__(self, trace:Optional[Trace], weights:Optional[Output], output:Optional[Output]):
        self.trace = trace
        self.weights = weights
        self.output = output
    def __repr__(self):
        if self.trace is None:
            return "None"
        else:
            out_str = tensor_utils.show(self.output) if isinstance(self.output, Tensor) else self.output
            return "tr={}; lv={}; out={}".format(trace_utils.show(self.trace), tensor_utils.show(self.weights) if self.weights is not None else "None", out_str)
    def __iter__(self):
        for x in [self.trace, self.weights, self.output]:
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

class Condition(Inf):
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    FIXME: can't condition a conditioned model at the moment
    """
    def __init__(self, process: Conditionable, trace: Optional[Trace], requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), _step=None) -> None:
        self.process = process
        self.conditioning_trace = trace_utils.copytrace(trace, requires_grad=requires_grad, detach=detach) if trace is not None else Trace()
        self._requires_grad = requires_grad
        self._detach = detach

    def __call__(self, *args:Any, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        self.process._cond_trace = self.conditioning_trace
        out = self.process(*args, **kwargs)
        self.process._cond_trace = Trace()
        return out

class KernelInf(nn.Module, Conditionable):
    def __init__(self, _step:Optional[int]=None, _permissive_arguments:bool=True):
        nn.Module.__init__(self)
        Conditionable.__init__(self)
        self._cache = KCache(None, None)
        self._step = _step
        self._permissive_arguments = _permissive_arguments

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))

    def _program_args(self, fn, *args):
        if self._permissive_arguments:
            assert args is None or len(args) == 0, "need to filter this list, but currently don't have an example"
            # return [v for k,v in args.items() if check_passable_arg(k, fn)]
            return args
        else:
            return args

    def _program_kwargs(self, fn, **kwargs):
        if self._permissive_arguments and isinstance(fn, Program):
            return {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.model)}
        else:
            return kwargs

    def _run_program(self, program, *program_args:Any, **program_kwargs:Any):
        # runnable = Condition(self.program, self._cond_trace) if self._cond_trace is not None else self.program
        return program(
            *self._program_args(program, *program_args),
            **self._program_kwargs(program, **program_kwargs))

    def _run_kernel(self, kernel, program_trace: Trace, program_output:Output, sample_dims=None):
        return kernel(program_trace, program_output, sample_dims=sample_dims)


class Reverse(KernelInf, Inf):
    def __init__(self, program: Union[Program, KernelInf], kernel: Kernel, _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs:Any) -> Tuple[Trace, Output]:
        program = Condition(self.program, self._cond_trace) if self._cond_trace is not None else self.program
        program_state = State(*self._run_program(program, *program_args, sample_dims=sample_dims, **program_kwargs))
        kernel_state = State(*self._run_kernel(self.kernel, program_state.trace, program_state.output, sample_dims=sample_dims))

        log_aux = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims)

        self._cache = KCache(program_state, kernel_state)

        return program_state.trace, log_aux, program_state.output

class Forward(KernelInf, Inf):
    def __init__(self, kernel: Kernel, program: Union[Program, KernelInf], _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs) -> Tuple[Trace, Optional[Tensor], Output]:
        program_state = State(*self._run_program(self.program, *program_args, sample_dims=sample_dims, **program_kwargs))
        kernel_state = State(*self._run_kernel(self.kernel, program_state.trace, program_state.output, sample_dims=sample_dims))
        log_joint = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims)

        self._cache = KCache(program_state, kernel_state)

        return kernel_state.trace, log_joint, kernel_state.output


class Propose(nn.Module, Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Program, Inf], validate:bool=True, _step=None):
        super().__init__()
        self.target = target
        self.proposal = proposal
        self._cache = PCache(None, None)
        self.validate = validate
        self._step = _step # used for debugging

    def forward(self, *shared_args, sample_dims=None, **shared_kwargs):
        proposal_state = State(*self.proposal(*shared_args, sample_dims=sample_dims, **shared_kwargs))

        conditioned_target = Condition(self.target, proposal_state.trace) # NOTE: might be a bug and needs whole trace?
        target_state = State(*conditioned_target(*shared_args, sample_dims=sample_dims, **shared_kwargs))
        lv = target_state.weights - proposal_state.weights


        self._cache = PCache(target_state, proposal_state)
        state = self._cache


        if self.proposal._cache is not None:
            # FIXME: this is a hack for the moment and should be removed somehow.
            # NOTE: this is unnecessary for the e2e/test_1dgaussians.py, but I am a bit nervous about double-gradients
            if isinstance(self.proposal._cache, PCache):
                raise NotImplemented("If this is the case (which it can be) i will actually need a way to propogate these detachments in a smarter way")

            joint_proposal_trace = self.proposal._cache.kernel.trace
            joint_target_trace = self.target._cache.kernel.trace
            # can we always assume we are in NVI territory?
            for k, rv in self.proposal._cache.program.trace.items():
                joint_proposal_trace[k]._value = rv.value.detach()

            proposal_keys = set(self.proposal._cache.program.trace.keys())
            target_keys = set(self.target._cache.kernel.trace.keys()) - proposal_keys
            state = PCache(
                proposal=State(trace=trace_utils.copysubtrace(proposal_state.trace, proposal_keys), weights=proposal_state.weights, output=proposal_state.output),
                target=State(trace=trace_utils.copysubtrace(target_state.trace, target_keys), weights=target_state.weights, output=target_state.output),
            )

        return state, lv
