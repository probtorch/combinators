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

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs, Out
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


class Inf:
    pass


def _dispatch(permissive):
    def curry_fn(fn):
        def go(*args:Any, **kwargs:Any):
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
    return curry_fn

class Condition(Inf):
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    FIXME: can't condition a conditioned model at the moment
    """
    def __init__(self, process: Conditionable, trace: Optional[Trace]=None, program:Optional[Inf]=None, requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), as_trace=True, full_trace_return=True, _step=None, _debug=False) -> None:
        assert trace is not None or program is not None
        self.process = process
        if trace is not None:
            self.conditioning_trace = trace_utils.copytrace(trace, requires_grad=requires_grad, detach=detach) if trace is not None else Trace()
        else:
            self.conditioning_program = _dispatch(permissive=True)(program)

        self._requires_grad = requires_grad
        self._detach = detach
        self._debug = _debug
        self.as_trace = as_trace
        self.full_trace_return = full_trace_return

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        conditioning_trace = self.conditioning_trace if self.conditioning_trace is not None else self.conditioning_program(*args, _debug=_debug, **kwargs)
        self.process._cond_trace = ConditioningTrace(conditioning_trace)

        out = _dispatch(permissive=True)(self.process)(*args, **kwargs)
        self.process._cond_trace = Trace()
        trace = out.trace
        if self.as_trace and isinstance(trace, ConditioningTrace):
            return Out(trace.as_trace(access_only=not self.full_trace_return), out.weights, out.output, extras=dict(type=type(self).__name__ + "(" + type(self.process).__name__ + ")"))
        else:
            return out

class KernelInf(Conditionable):
    def __init__(self, _step:Optional[int]=None, _permissive_arguments:bool=True):
        nn.Module.__init__(self)
        Conditionable.__init__(self)
        self._cache = Out(None, None, None)
        self._step = _step
        self._permissive_arguments = _permissive_arguments

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))


class Reverse(KernelInf, Inf):
    def __init__(self, program: Union[Program, KernelInf], kernel: Kernel, loss_fn:Callable[[Tensor, Tensor], Tensor]=(lambda x, fin: fin), loss0=torch.zeros(1), _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel
        self.foldl_loss = loss_fn
        self.loss0 = loss0

    def __call__(self, *program_args:Any, sample_dims=None, _debug=False, **program_kwargs:Any) -> Tuple[Trace, Optional[Tensor], Output]:
        program = Condition(self.program, trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.program

        program_state = _dispatch(permissive=True)(program)(*program_args, sample_dims=sample_dims, **program_kwargs)

        kernel = Condition(self.kernel, trace=self._cond_trace) if self._cond_trace is not None else self.kernel
        kernel_state = _dispatch(permissive=True)(kernel)(program_state.trace, program_state.output, sample_dims=sample_dims)

        log_aux = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims)

        out_trace = program_state.trace.as_trace(access_only=True) if isinstance(program_state.trace, ConditioningTrace) \
                        else program_state.trace

        self._cache = Out(
            trace=out_trace,
            weights=log_aux,
            output=program_state.output,
            extras=dict(
                program=program_state,
                kernel=kernel_state,
                type=type(self).__name__,
                loss=self.foldl_loss(log_aux, kernel_state.loss if 'loss' in kernel_state else self.loss0)))

        return self._cache

class Forward(KernelInf, Inf):
    def __init__(self, kernel: Kernel, program: Union[Program, KernelInf], loss_fn:Callable[[Tensor, Tensor], Tensor]=(lambda x, fin: fin), loss0=torch.zeros(1), _step=None, _permissive=True) -> None:
        super().__init__(_step, _permissive)
        self.program = program
        self.kernel = kernel
        self.foldl_loss = loss_fn
        self.loss0 = loss0
        self._run_program = _dispatch(permissive=True)(self.program)
        self._run_kernel = _dispatch(permissive=True)(self.kernel)

    def __call__(self, *program_args:Any, sample_dims=None, _debug=False, **program_kwargs) -> Tuple[Trace, Optional[Tensor], Output]:
        program_state = self._run_program(*program_args, sample_dims=sample_dims, **program_kwargs)

        kernel_state = self._run_kernel(program_state.trace, program_state.output, sample_dims=sample_dims)
        log_joint = kernel_state.trace.log_joint(batch_dim=None, sample_dims=sample_dims)

        self._cache = Out(
            trace=kernel_state.trace,
            weights=log_joint,
            output=kernel_state.output,
            extras=dict(
                program=program_state,
                kernel=kernel_state,
                type=type(self).__name__,
                loss=self.foldl_loss(log_joint, kernel_state.loss if 'loss' in kernel_state else self.loss0)))

        return self._cache


class Propose(Inf):
    def __init__(self, target: Union[Program, KernelInf], proposal: Union[Program, Inf], loss_fn:Callable[[Tensor, Tensor], Tensor]=(lambda x, fin: fin), loss0=torch.zeros(1), _step:Optional[int]=None, _debug:bool=False):
        super().__init__()
        self.target = target
        self.proposal = proposal
        self._cache = Out(None, None, None)
        self.foldl_loss = loss_fn
        self._step = _step # used for debugging
        self._debug = _debug
        self.loss0 = loss0

    def __call__(self, *shared_args, sample_dims=None, _debug=False, **shared_kwargs):
        proposal_state = self.proposal(*shared_args, sample_dims=sample_dims, **shared_kwargs)

        conditioned_target = Condition(self.target, trace=proposal_state.trace, requires_grad=RequiresGrad.YES) # NOTE: might be a bug and needs whole trace?
        target_state = conditioned_target(*shared_args, sample_dims=sample_dims, **shared_kwargs)
        lv = target_state.weights - proposal_state.weights

        self._cache = Out(
            extras=dict(
                proposal=proposal_state if self._debug or _debug else Out(*proposal_state), # strip auxiliary traces
                target=target_state if self._debug  or _debug else Out(*target_state), # strip auxiliary traces
                type=type(self).__name__,
                loss=self.foldl_loss(lv, proposal_state.loss if 'loss' in proposal_state else self.loss0)
                ),
            trace=target_state.trace,
            weights=lv,
            output=target_state.output,)

        return self._cache
