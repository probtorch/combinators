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
from combinators.stochastic import Trace, ConditioningTrace
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


def _dispatch(permissive):
    def go(fn, *args:Any, **kwargs:Any):
        if isinstance(fn, Program):
            spec_fn = fn.model
        elif isinstance(fn, Kernel):
            spec_fn = fn.apply_kernel
        else:
            spec_fn = fn

        _dispatch_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, spec_fn)} if permissive else kwargs
        _dispatch_args   = args
        # _dispatch_args   = [v for k,v in args.items() if check_passable_arg(k, fn)] if permissive else args
        # assert args is None or len(args) == 0, "need to filter this list, but currently don't have an example"

        return fn(*_dispatch_args, **_dispatch_kwargs)
    return go

class Condition(Inf):
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    FIXME: can't condition a conditioned model at the moment
    """
    def __init__(self, process: Conditionable, trace: Optional[Trace], requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), _step=None, _debug=False) -> None:
        self.process = process
        self.conditioning_trace = trace_utils.copytrace(trace, requires_grad=requires_grad, detach=detach) if trace is not None else Trace()
        self._requires_grad = requires_grad
        self._detach = detach
        self._debug = _debug

    def __call__(self, *args:Any, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        self.process._cond_trace = ConditioningTrace(self.conditioning_trace)

        out = _dispatch(permissive=True)(self.process, *args, **kwargs)
        self.process._cond_trace = Trace()
        trace = out[0]
        if isinstance(trace, ConditioningTrace):
            return out[0].as_trace(access_only=True), *out[1:]
        else:
            return out

class KernelInf(nn.Module, Conditionable):
    def __init__(self, _step:Optional[int]=None, _permissive_arguments:bool=True):
        nn.Module.__init__(self)
        Conditionable.__init__(self)
        self._cache = KCache(None, None)
        self._step = _step
        self._permissive_arguments = _permissive_arguments
        self._run_program = _dispatch(permissive=True)
        self._run_kernel = _dispatch(permissive=True)

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))


class Reverse(KernelInf, Inf):
    def __init__(self, program: Union[Program, KernelInf], kernel: Kernel, _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel

    def forward(self, *program_args:Any, sample_dims=None, **program_kwargs:Any) -> Tuple[Trace, Optional[Tensor], Output]:
        program = Condition(self.program, self._cond_trace, _debug=True) if self._cond_trace is not None else self.program

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

        conditioned_target = Condition(self.target, proposal_state.trace, requires_grad=RequiresGrad.YES) # NOTE: might be a bug and needs whole trace?
        target_state = State(*conditioned_target(*shared_args, sample_dims=sample_dims, **shared_kwargs))
        lv = target_state.weights - proposal_state.weights


        self._cache = PCache(target_state, proposal_state)
        state = self._cache



        return state, lv
