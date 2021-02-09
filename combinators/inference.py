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
from combinators.stochastic import Trace, ConditioningTrace, Provenance
from combinators.program import Program
from combinators.kernel import Kernel
from combinators.traceable import Conditionable
from combinators.objectives import nvo_avo

def maybe(obj, name, default, fn=(lambda x: x)):
    return fn(getattr(obj, name)) if hasattr(obj, name) else default

def _dispatch():
    def get_callable(fn):
        if isinstance(fn, Program):
            spec_fn = fn.model
        elif isinstance(fn, Kernel):
            spec_fn = fn.apply_kernel
        else:
            spec_fn = fn
        return spec_fn
    return dispatch(get_callable, permissive=True)

class Inf(ABC):
    def __init__(
            self,
            loss_fn:Callable[[Out, Tensor], Tensor]=(lambda _, fin: fin),
            loss0=None,
            device=None,
            ix:Union[Tuple[int], NamedTuple, None]=None,
            _debug=False,
            sample_dims=None,
            batch_dim=None):
        self.loss0 = torch.zeros(1, device=autodevice(device)) if loss0 is None else loss0
        self.foldr_loss = loss_fn
        self.ix = ix
        self._debug = _debug
        self._out = Out(None, None, None)
        self.batch_dim = batch_dim
        self.sample_dims = sample_dims

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
            cond_trace: Optional[Trace]=None,
            # program: Optional[Inf]=None,
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
        assert cond_trace is not None # or program is not None
        self.process = process

        if requires_grad == RequiresGrad.NO:
            self.conditioning_trace = trace_utils.copytrace(cond_trace, requires_grad=requires_grad, detach=detach)
        else:
        # FIXME: do we actually need a copy of the trace?
            self.conditioning_trace = cond_trace

        self._requires_grad = requires_grad
        self._detach = detach
        self.as_trace = as_trace
        self.full_trace_return = full_trace_return

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        """ Condition """
        extras=dict(type=type(self).__name__ + "(" + type(self.process).__name__ + ")")

        self.process._cond_trace = ConditioningTrace(self.conditioning_trace)

        out = _dispatch()(self.process)(*args, **kwargs)

        self.process._cond_trace = Trace()

        trace = out.trace
        for k, v in out.items():
            if k not in ['conditioned_output', 'trace', 'log_weight', 'output']:
                extras[k] = v

        if self.as_trace and isinstance(trace, ConditioningTrace):
            return Out(trace.as_trace(access_only=not self.full_trace_return), out.log_weight, out.output, extras=extras)
        else:
            return out

class Resample(Inf):
    """
    Compute importance weight of the proposal program's trace under the target program's trace,
    considering the incomming log weight lw of the proposal's trace
    """
    def __init__(
            self,
            program: Union[Program, Inf],
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

        program_out = self.program(*shared_args, **inf_kwargs, **shared_kwargs)

        passable_kwargs = {k: v for k, v in shape_kwargs.items() if check_passable_kwarg(k, self.strategy)}

        tr_, lw_ = self.strategy(program_out.trace, program_out.log_weight, **passable_kwargs)

        self._out = Out(
            extras=dict(
                program_out=program_out,
                type=type(self).__name__,
                ix=ix,
                ),
            trace=tr_,
            log_weight=lw_,
            output=program_out.output)

        self._out['loss'] = self.foldr_loss(self._out, maybe(program_out, 'loss', self.loss0))

        return self._out

class KernelInf(Conditionable, Inf):
    def __init__(
            self,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False):
        Conditionable.__init__(self)
        Inf.__init__(self, ix=ix, _debug=_debug, loss_fn=loss_fn, loss0=loss0, device=device)

    def _show_traces(self):
        if all(map(lambda x: x is None, self._out)):
            print("No traces found!")
        else:
            print("program: {}".format(self._out.program.trace))
            print("kernel : {}".format(self._out.kernel.trace))


class Extend(KernelInf):
    def __init__(self,
            program: Program,
            kernel: Kernel,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False) -> None:
        super().__init__(loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.program = program
        self.kernel = kernel

    def __call__(self, *shared_args:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs:Any) -> Out:
        """ Extend """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix = self.ix if self.ix is not None else ix, **shape_kwargs)

        program = Condition(self.program, cond_trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.program

        program_out = _dispatch()(program)(*shared_args, **inf_kwargs, **shared_kwargs)

        kernel = Condition(self.kernel, cond_trace=self._cond_trace) if self._cond_trace is not None else self.kernel

        kernel_out = _dispatch()(kernel)(program_out.trace, program_out.output, **inf_kwargs, **shared_kwargs)
        # FIXME: when we get rid of kernels and put extend combinators here, kernel_out.log_weight must be 0 and we need an assert here
        assert len(set(kernel_out.trace.keys()).intersection(set(program_out.trace.keys()))) == 0

        log_joint_extended = kernel_out.trace.log_joint(**shape_kwargs, nodes={k for k,v in kernel_out.trace.items() if v.provenance != Provenance.OBSERVED})

        self._out = Out(
            trace=program_out.trace,
            log_weight=program_out.log_weight + log_joint_extended, # $w_1 \cdot u_2$
            output=program_out.output,
            extras=dict(
                program=program_out,
                kernel=kernel_out,
                type=type(self).__name__,
                ix=ix,
                ))

        self._out['loss'] = self.foldr_loss(self._out, maybe(program_out, 'loss', self.loss0))

        return self._out

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

        program_out = self._run_program(*program_args, **inf_kwargs, **program_kwargs)

        kernel_out = self._run_kernel(program_out.trace, program_out.output, **inf_kwargs, **program_kwargs)

        log_prob = kernel_out.trace.log_joint(**shape_kwargs, nodes=self.joint_set(kernel_out.trace))
        if isinstance(self.kernel, Program):
            log_prob += program_out.trace.log_joint(**shape_kwargs, nodes=self.joint_set(program_out.trace))

        self._out = Out(
            trace=kernel_out.trace,
            log_prob=log_prob,
            output=kernel_out.output,
            extras=dict(
                program=program_out,
                kernel=kernel_out,
                type=type(self).__name__,
                ix=ix,
                ))

        if ix is not None:
            self._out['ix'] = ix
        if 'log_cweight' in program_out:
            self._out['log_cweight'] = program_out['log_cweight']
        self._out['loss'] = self.foldr_loss(self._out, maybe(program_out, 'loss', self.loss0))
        return self._out


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
        proposal_out = self.proposal(*shared_args, **inf_kwargs, **shared_kwargs)

        # conditioned_target = Condition(self.target, trace=proposal_out.trace, requires_grad=RequiresGrad.YES) # NOTE: might be a bug and _doesn't_ need the whole trace?
        target_out = _dispatch(True)(self.target)(proposal_out.output, *shared_args, **inf_kwargs,  **shared_kwargs)

        # ppr(proposal_out.trace, desc="proposal trace from:")
        # ppr(target_out.kernel.trace if target_out.type == "Reverse" else target_out.trace, desc="target trace from:")

        lv = target_out.log_prob - proposal_out.log_prob
        # pprm(lv, name="log_w")
        # pprm(target_out.log_prob, name="log_p")
        # pprm(proposal_out.log_prob, name="log_p")
        # print("done")

        # if ix.t <= IX:
        #     print()
        # if ix.t <= IX:
        #     print("log_fwd {}\t{: .4f}\t".format(ix.t, proposal_out.log_prob.mean().item()), tensor_utils.show(proposal_out.log_prob))
        #     print("log_tar {}\t{: .4f}\t".format(ix.t, target_out.log_prob.mean().item()), tensor_utils.show(target_out.log_prob), )
        #     print("log_inc {}\t{: .4f}\t".format(ix.t, lv.mean().item()), tensor_utils.show(lv) )


        # if ix.t <= IX:
        #     ppr(proposal_out.trace, desc="   fwd_kernel.trace", delim="\n     ")
        #     ppr(target_out.trace,   desc="target_kernel.trace", delim="\n     ")
        self._out = Out(
            extras=dict(
                proposal=proposal_out if self._debug or _debug else Out(*proposal_out), # strip auxiliary traces
                target=target_out if self._debug  or _debug else Out(*target_out), # strip auxiliary traces
                type=type(self).__name__,

                # only way this happens if we are recursively defining propose statements.
                log_weight=lv,
                log_cweight=lv if 'log_cweight' not in proposal_out else lv + proposal_out.log_cweight,
                ix=ix,
                ),
            trace=target_out.trace,
            log_prob=target_out.log_prob,
            output=target_out.output,)

        if ix is not None:
            self._out['ix'] = ix

        self._out['loss'] = self.foldr_loss(self._out, maybe(proposal_out, 'loss', self.loss0))

        return self._out
