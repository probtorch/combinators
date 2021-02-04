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
from typing import Iterable, NamedTuple
import operator
from inspect import signature
import inspect

import combinators.debug as debug
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
import combinators.resampling.strategies as rstrat

from combinators.types import check_passable_arg, check_passable_kwarg, get_shape_kwargs, Out, Output, State, TraceLike
from combinators.utils import dispatch, ppr, pprm
from combinators.trace.utils import RequiresGrad, copytrace, mapvalues, disteq
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.stochastic import Trace, ConditioningTrace
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Conditionable
from combinators.objectives import nvo_avo

def maybe(obj, name, default, fn=(lambda x: x)):
    return fn(getattr(obj, name)) if hasattr(obj, name) else default

IX = -20

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
            loss_fn:Callable[[Out, Tensor], Tensor]=(lambda _, fin: fin),
            loss0=None,
            device=None,
            ix:Union[Tuple[int], NamedTuple, None]=None,
            _debug=False,
            sample_dim=None,
            exclude=None,
            include=None,
            batch_dim=None):
        self.loss0 = torch.zeros(1, device=autodevice(device)) if loss0 is None else loss0
        self.foldr_loss = loss_fn
        self.ix = ix
        self._debug = _debug
        self._cache = Out(None, None, None)
        self.batch_dim = batch_dim
        self.sample_dim = sample_dim
        assert not (exclude is not None and include is not None)
        self.exclude = exclude
        self.include = include

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        raise NotImplementedError("@abstractproperty but type system doesn't understand it")

    def joint_set(self, trace):
        tnodes = set(trace.keys())
        if self.exclude is not None:
            nodes = tnodes - self.exclude
        elif self.include is not None:
            assert len(self.include - tnodes) == 0
            nodes = self.include
        else:
            nodes = None
        return nodes


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
            ix=None,
            _debug=False,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None) -> None:
        Inf.__init__(self, ix=ix, _debug=_debug, loss_fn=loss_fn, loss0=loss0, device=device)
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

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        """ Condition """
        extras=dict(type=type(self).__name__ + "(" + type(self.process).__name__ + ")")
        if self.conditioning_trace is not None:
            conditioning_trace = self.conditioning_trace
        else:
            cprog_out = self.conditioning_program(*args, _debug=_debug, ix=self.ix, **kwargs)
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
            return Out(trace.as_trace(access_only=not self.full_trace_return), out.log_prob, out.output, extras=extras)
        else:
            return out

class Resample(Inf):
    """
    Compute importance weight of the proposal program's trace under the target program's trace,
    considering the incomming log weight lw of the proposal's trace
    """
    def __init__(
            self,
            program: Inf, # not Union[Program, Inf] because (@stites) is not computing log_joint for programs, (which I guess would be the joint?)
            ix=None,
            _debug:bool=False,
            loss0=None,
            strategy=rstrat.Systematic()):
        super().__init__(ix=ix, _debug=_debug, loss0=loss0)
        self.program = program
        self.strategy = strategy

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Resample """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix=self.ix if self.ix is not None else ix, **shape_kwargs)

        program_state = self.program(*shared_args, **inf_kwargs, **shared_kwargs)
        # change_tr = mapvalues(program_state.trace, mapper=lambda v: v.unsqueeze(1))
        # change_lw = program_state.log_weight.unsqueeze(1)
        # tr, lw = program_state.trace, program_state.log_weight # aggregating state is currently a PITA
        rkwargs = dict(sample_dim=sample_dims, batch_dim=batch_dim)

        passable_kwargs = {k: v for k, v in rkwargs.items() if check_passable_kwarg(k, self.strategy)}

        tr_, lw_ = self.strategy(program_state.trace, program_state.log_prob, **passable_kwargs)

        self._cache = Out(
            extras=dict(
                program=program_state if self._debug or _debug else Out(*program_state), # strip auxiliary traces
                type=type(self).__name__,
                log_weight=lw_,
                ix=ix,
                ),
            trace=tr_,
            log_prob=lw_,
            output=program_state.output)
        self._cache['loss'] = self.foldr_loss(self._cache, maybe(program_state, 'loss', self.loss0))

        return self._cache

class KernelInf(Conditionable, Inf):
    def __init__(self,
            _permissive_arguments:bool=True,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            exclude=None,
            include=None,
            _debug=False):
        Conditionable.__init__(self)
        Inf.__init__(self, ix=ix, _debug=_debug, loss_fn=loss_fn, loss0=loss0, device=device, exclude=exclude, include=include)
        self._permissive_arguments = _permissive_arguments

    def _show_traces(self):
        if all(map(lambda x: x is None, self._cache)):
            print("No traces found!")
        else:
            print("program: {}".format(self._cache.program.trace))
            print("kernel : {}".format(self._cache.kernel.trace))


class Reverse(KernelInf):
    def __init__(self,
            program: Union[Program, KernelInf, Condition, Resample], # anything that isn't a propose
            kernel: Kernel,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False,
            exclude=None,
            include=None,
            _permissive=True) -> None:
        super().__init__(_permissive, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug, exclude=exclude, include=include)
        self.program = program
        self.kernel = kernel

    def __call__(self, *program_args:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **program_kwargs:Any) -> Out:
        """ Reverse """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        ix = self.ix if self.ix is not None else ix
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        program = Condition(self.program, trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.program

        program_state = _dispatch(permissive=True)(program)(*program_args, **inf_kwargs, **program_kwargs)

        kernel = Condition(self.kernel, trace=self._cond_trace) if self._cond_trace is not None else self.kernel

        kernel_state = _dispatch(permissive=True)(kernel)(program_state.trace, program_state.output, **inf_kwargs, **program_kwargs)

        log_aux = kernel_state.trace.log_joint(**shape_kwargs, nodes=self.joint_set(kernel_state.trace))

        out_trace = program_state.trace.as_trace(access_only=True) if isinstance(program_state.trace, ConditioningTrace) \
                        else program_state.trace

        self._cache = Out(
            trace=out_trace,
            log_prob=log_aux,
            output=program_state.output,
            extras=dict(
                program=program_state,
                kernel=kernel_state,
                type=type(self).__name__,
                log_weight=maybe(program_state, 'log_weight', 0),
                ix=ix,
                ))

        if 'log_cweight' in program_state:
            self._cache['log_cweight'] = program_state['log_cweight']
        if ix is not None:
            self._cache['ix'] = ix
        self._cache['loss'] = self.foldr_loss(self._cache, maybe(program_state, 'loss', self.loss0))

        return self._cache

class Forward(KernelInf):
    def __init__(
            self,
            kernel: Union[Program, Kernel],
            program: Union[Program, Condition, Resample, KernelInf, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False,
            _permissive=True,
            exclude=None,
            include=None,
    ) -> None:
        super().__init__(_permissive, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug, exclude=exclude, include=include)
        self.program = program
        self.kernel = kernel
        self._run_program = _dispatch(permissive=True)(self.program)
        self._run_kernel = _dispatch(permissive=True)(self.kernel)

    def __call__(self, *program_args:Any, sample_dims=None, batch_dim=None, _debug=False, _debug_extras=None, reparameterized=True, ix=None, **program_kwargs) -> Out:
        """ Forward """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        ix = self.ix if self.ix is not None else ix
        inf_kwargs = dict(_debug=_debug, ix=ix, **shape_kwargs)

        program_state = self._run_program(*program_args, **inf_kwargs, **program_kwargs)

        kernel_state = self._run_kernel(program_state.trace, program_state.output, **inf_kwargs, **program_kwargs)

        log_prob = kernel_state.trace.log_joint(**shape_kwargs, nodes=self.joint_set(kernel_state.trace))
        if isinstance(self.kernel, Program):
            log_prob += program_state.trace.log_joint(**shape_kwargs, nodes=self.joint_set(program_state.trace))

        self._cache = Out(
            trace=kernel_state.trace,
            log_prob=log_prob,
            output=kernel_state.output,
            extras=dict(
                program=program_state,
                kernel=kernel_state,
                type=type(self).__name__,
                ix=ix,
                ))

        if ix is not None:
            self._cache['ix'] = ix
        if 'log_cweight' in program_state:
            self._cache['log_cweight'] = program_state['log_cweight']
        self._cache['loss'] = self.foldr_loss(self._cache, maybe(program_state, 'loss', self.loss0))
        return self._cache


class Propose(Conditionable, Inf):
    def __init__(self,
            target: Union[Program, KernelInf],
            proposal: Union[Program, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug:bool=False):
        Conditionable.__init__(self)
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.target = target
        self.proposal = proposal

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, _debug_extras=None, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Proposal """
        ix = self.ix if self.ix is not None else ix
        inf_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized, _debug=_debug, ix=ix)

        # proposal = Condition(self.proposal, trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.proposal
        proposal_state = self.proposal(*shared_args, **inf_kwargs, **shared_kwargs)

        # conditioned_target = Condition(self.target, trace=proposal_state.trace, requires_grad=RequiresGrad.YES) # NOTE: might be a bug and _doesn't_ need the whole trace?
        target_state = _dispatch(True)(self.target)(proposal_state.output, *shared_args, **inf_kwargs,  **shared_kwargs)

        # ppr(proposal_state.trace, desc="proposal trace from:")
        # ppr(target_state.kernel.trace if target_state.type == "Reverse" else target_state.trace, desc="target trace from:")

        lv = target_state.log_prob - proposal_state.log_prob
        # pprm(lv, name="log_w")
        # pprm(target_state.log_prob, name="log_p")
        # pprm(proposal_state.log_prob, name="log_p")
        # print("done")

        # if ix.t <= IX:
        #     print()
        # if ix.t <= IX:
        #     print("log_fwd {}\t{: .4f}\t".format(ix.t, proposal_state.log_prob.mean().item()), tensor_utils.show(proposal_state.log_prob))
        #     print("log_tar {}\t{: .4f}\t".format(ix.t, target_state.log_prob.mean().item()), tensor_utils.show(target_state.log_prob), )
        #     print("log_inc {}\t{: .4f}\t".format(ix.t, lv.mean().item()), tensor_utils.show(lv) )


        # if ix.t <= IX:
        #     ppr(proposal_state.trace, desc="   fwd_kernel.trace", delim="\n     ")
        #     ppr(target_state.trace,   desc="target_kernel.trace", delim="\n     ")
        self._cache = Out(
            extras=dict(
                proposal=proposal_state if self._debug or _debug else Out(*proposal_state), # strip auxiliary traces
                target=target_state if self._debug  or _debug else Out(*target_state), # strip auxiliary traces
                type=type(self).__name__,

                # only way this happens if we are recursively defining propose statements.
                log_weight=lv,
                log_cweight=lv if 'log_cweight' not in proposal_state else lv + proposal_state.log_cweight,
                ix=ix,
                ),
            trace=target_state.trace,
            log_prob=target_state.log_prob,
            output=target_state.output,)

        if ix is not None:
            self._cache['ix'] = ix

        self._cache['loss'] = self.foldr_loss(self._cache, maybe(proposal_state, 'loss', self.loss0))

        return self._cache
