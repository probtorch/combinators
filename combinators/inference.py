#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor, distributions
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod, abstractproperty
import inspect
import ast
import weakref
from typing import Iterable
import operator

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs, Out
from combinators.utils import dispatch
from combinators.trace.utils import RequiresGrad, copytrace, mapvalues
from combinators.tensor.utils import autodevice, kw_autodevice
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
import combinators.resampling.strategies as rstrat

def maybe(obj, name, default, fn=(lambda x: x)):
    return fn(getattr(obj, name)) if hasattr(obj, name) else default



def _dispatch(permissive):
    def get_callable(fn):
        if isinstance(fn, Program):
            spec_fn = fn.model
        elif isinstance(fn, Kernel):
            spec_fn = fn.apply_kernel
        else:
            spec_fn = fn
        return spec_fn
    return dispatch(get_callable, permissive)

class Inf(ABC):
    def __init__(
            self,
            loss_fn:Callable[[Tensor, Tensor], Tensor]=(lambda _, fin: fin),
            loss0=None,
            device=None,
            _step=None,
            _debug=False,
            sample_dim=None,
            batch_dim=None):
        self.loss0 = torch.zeros(1, device=autodevice(device)) if loss0 is None else loss0.to(autodevice(device))
        self.foldl_loss = loss_fn
        self._step = _step
        self._debug = _debug
        self._cache = Out(None, None, None)
        self.batch_dim = batch_dim
        self.sample_dim = sample_dim

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        raise NotImplementedError("@abstractproperty but type system doesn't understand it")


class Condition(Inf):
    """
    Run a program's model with a conditioned trace
    TOOO: should also be able to Condition any combinator.
    FIXME: can't condition a conditioned model at the moment
    """
    def __init__(self,
            process: Conditionable,
            trace: Optional[Trace]=None,
            program:Optional[Inf]=None,
            requires_grad:RequiresGrad=RequiresGrad.DEFAULT,
            detach:Set[str]=set(),
            as_trace=True,
            full_trace_return=True,
            _step=None,
            _debug=False,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None) -> None:
        Inf.__init__(self, _step=_step, _debug=_debug, loss_fn=loss_fn, loss0=loss0, device=device)
        assert trace is not None or program is not None
        self.process = process
        if trace is not None:
            self.conditioning_trace = trace_utils.copytrace(trace, requires_grad=requires_grad, detach=detach) if trace is not None else Trace()
        else:
            self.conditioning_program = _dispatch(permissive=True)(program)

        self._requires_grad = requires_grad
        self._detach = detach
        self.as_trace = as_trace
        self.full_trace_return = full_trace_return

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Tuple[Trace, Optional[Trace], Output]:
        extras=dict(type=type(self).__name__ + "(" + type(self.process).__name__ + ")")
        if self.conditioning_trace is not None:
            conditioning_trace = self.conditioning_trace
        else:
            cprog_out = self.conditioning_program(*args, _debug=_debug, **kwargs)
            conditioning_trace = cprog_out.trace
            extras["conditioned_output"] = cprog_out
            raise RuntimeError("[wip] requires testing")

        self.process._cond_trace = ConditioningTrace(conditioning_trace)

        out = _dispatch(permissive=True)(self.process)(*args, **kwargs)
        self.process._cond_trace = Trace()
        trace = out.trace
        for k, v in out.items():
            if k not in ['conditioned_output', 'trace', 'weight', 'output']:
                extras[k] = v
        if self.as_trace and isinstance(trace, ConditioningTrace):
            return Out(trace.as_trace(access_only=not self.full_trace_return), out.weights, out.output, extras=extras)
        else:
            return out

class KernelInf(Conditionable, Inf):
    def __init__(self,
            _permissive_arguments:bool=True,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            _step=None,
            _debug=False):
        Conditionable.__init__(self)
        Inf.__init__(self, _step=_step, _debug=_debug, loss_fn=loss_fn, loss0=loss0, device=device)
        self._permissive_arguments = _permissive_arguments

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))


class Reverse(KernelInf):
    def __init__(self,
            program: Union[Program, KernelInf],
            kernel: Kernel,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            _step=None,
            _debug=False,
            _permissive=True) -> None:
        super().__init__(_permissive, loss_fn=loss_fn, loss0=loss0, device=device, _step=_step, _debug=_debug)
        self.program = program
        self.kernel = kernel

    def __call__(self, *program_args:Any, sample_dims=None, batch_dim=None, _debug=False, **program_kwargs:Any) -> Tuple[Trace, Optional[Tensor], Output]:
        program = Condition(self.program, trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.program

        program_state = _dispatch(permissive=True)(program)(*program_args, sample_dims=sample_dims, batch_dim=batch_dim, **program_kwargs)

        kernel = Condition(self.kernel, trace=self._cond_trace) if self._cond_trace is not None else self.kernel
        kernel_state = _dispatch(permissive=True)(kernel)(program_state.trace, program_state.output, sample_dims=sample_dims, batch_dim=batch_dim)

        log_aux = kernel_state.trace.log_joint(sample_dims=sample_dims, batch_dim=batch_dim)

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
                loss=self.foldl_loss(log_aux, maybe(kernel_state, 'loss', self.loss0)),
                cumulative_log_weight=maybe(program_state, 'cumulative_log_weight', 0),
                ))

        return self._cache

class Forward(KernelInf):
    def __init__(self,
            kernel: Kernel,
            program: Union[Program, KernelInf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            _step=None,
            _debug=False,
            _permissive=True) -> None:
        super().__init__(_permissive, loss_fn=loss_fn, loss0=loss0, device=device, _step=_step, _debug=_debug)
        self.program = program
        self.kernel = kernel
        self._run_program = _dispatch(permissive=True)(self.program)
        self._run_kernel = _dispatch(permissive=True)(self.kernel)

    def __call__(self, *program_args:Any, sample_dims=None, batch_dim=None, _debug=False, **program_kwargs) -> Tuple[Trace, Optional[Tensor], Output]:
        program_state = self._run_program(*program_args, sample_dims=sample_dims, batch_dim=batch_dim, **program_kwargs)

        kernel_state = self._run_kernel(program_state.trace, program_state.output, sample_dims=sample_dims, batch_dim=batch_dim)
        log_joint = kernel_state.trace.log_joint(sample_dims=sample_dims, batch_dim=batch_dim)

        self._cache = Out(
            trace=kernel_state.trace,
            weights=log_joint,
            output=kernel_state.output,
            extras=dict(
                program=program_state,
                kernel=kernel_state,
                type=type(self).__name__,
                loss=self.foldl_loss(log_joint, maybe(kernel_state, 'loss', self.loss0)),
                cumulative_log_weight= maybe(program_state, 'cumulative_log_weight', 0),
                ))

        return self._cache


class Propose(Inf):
    def __init__(self,
            target: Union[Program, KernelInf],
            proposal: Union[Program, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            _step:Optional[int]=None,
            _debug:bool=False):
        super().__init__(loss_fn=loss_fn, loss0=loss0, device=device, _step=_step, _debug=_debug)
        self.target = target
        self.proposal = proposal

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, **shared_kwargs):
        proposal_state = self.proposal(*shared_args, sample_dims=sample_dims, batch_dim=batch_dim, **shared_kwargs)

        conditioned_target = Condition(self.target, trace=proposal_state.trace, requires_grad=RequiresGrad.YES) # NOTE: might be a bug and _doesn't_ need the whole trace?
        target_state = conditioned_target(*shared_args, sample_dims=sample_dims, batch_dim=batch_dim, **shared_kwargs)
        lv = target_state.weights - proposal_state.weights

        self._cache = Out(
            extras=dict(
                proposal=proposal_state if self._debug or _debug else Out(*proposal_state), # strip auxiliary traces
                target=target_state if self._debug  or _debug else Out(*target_state), # strip auxiliary traces
                type=type(self).__name__,
                loss=self.foldl_loss(lv, maybe(proposal_state, 'loss', self.loss0)),
                cumulative_log_weight=maybe(proposal_state, 'cumulative_log_weight', lv, lambda lw: lw+lv),
                ),
            trace=target_state.trace,
            weights=lv,
            output=target_state.output,)

        return self._cache

class Resample(Inf):
    """
    Compute importance weight of the proposal program's trace under the target program's trace,
    considering the incomming log weight lw of the proposal's trace
    """
    def __init__(
            self,
            program: Inf, # not Union[Program, Inf] because (@stites) is not computing weights for programs, (which I guess would be the joint?)
            _step:Optional[int]=None,
            _debug:bool=False,
            strategy=rstrat.Systematic):
        super().__init__(_step=_step, _debug=_debug)
        self.program = program
        self.strategy = strategy()

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, **shared_kwargs):
        program_state = self.program(*shared_args, sample_dims=sample_dims, batch_dim=batch_dim, **shared_kwargs)
        # change_tr = mapvalues(program_state.trace, mapper=lambda v: v.unsqueeze(1))
        # change_lw = program_state.cumulative_log_weight.unsqueeze(1)
        # tr, lw = program_state.trace, program_state.cumulative_log_weight # aggregating state is currently a PITA

        tr_, lw_ = self.strategy(program_state.trace, program_state.cumulative_log_weight, sample_dim=sample_dims, batch_dim=batch_dim)

        self._cache = Out(
            extras=dict(
                program=program_state if self._debug or _debug else Out(*program_state), # strip auxiliary traces
                type=type(self).__name__,
                cumulative_log_weight=lw_,
                ),
            trace=tr_,
            weights=lw_,
            output=program_state.output)

        return self._cache
