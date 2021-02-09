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
            program: Conditionable,
            cond_trace: Optional[Trace]=None,
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
        self.program = program

        # FIXME: do we actually need a copy of the trace?
        self.conditioning_trace = trace_utils.copytrace(cond_trace, requires_grad=requires_grad, detach=detach) \
            if requires_grad == RequiresGrad.NO else cond_trace

        self._requires_grad = requires_grad
        self._detach = detach
        self.as_trace = as_trace
        self.full_trace_return = full_trace_return

    def __call__(self, *args:Any, _debug=False, **kwargs:Any) -> Out:
        """ Condition """
        extras=dict(type=type(self).__name__ + "(" + type(self.program).__name__ + ")")

        self.program._cond_trace = ConditioningTrace(self.conditioning_trace)

        out = _dispatch()(self.program)(*args, **kwargs)

        self.program._cond_trace = Trace()

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
            q: Union[Program, Inf],
            ix=None,
            _debug:bool=False,
            loss0=None,
            strategy=rstrat.Systematic()):
        super().__init__(ix=ix, _debug=_debug, loss0=loss0)
        self.q = q
        self.strategy = strategy

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Resample """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix=self.ix if self.ix is not None else ix, **shape_kwargs)

        q_out = self.q(*shared_args, **inf_kwargs, **shared_kwargs)

        passable_kwargs = {k: v for k, v in shape_kwargs.items() if check_passable_kwarg(k, self.strategy)}

        tr_2, lw_2 = self.strategy(q_out.trace, q_out.log_weight, **passable_kwargs)

        self._out = Out(
            extras=dict(
                q_out=q_out,
                type=type(self).__name__,
                ix=ix,
                ),
            trace=tr_2,
            log_weight=lw_2,
            output=q_out.output)

        self._out['loss'] = self.foldr_loss(self._out, maybe(q_out, 'loss', self.loss0))

        return self._out


class Extend(Inf, Conditionable):
    def __init__(self,
            p: Program,
            f: Program, # FIXME: make this :=  f | extend (p, q) later
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False) -> None:
        Conditionable.__init__(self)
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.p = p
        self.f = f

    def __call__(self, *shared_args:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs:Any) -> Out:
        """ Extend """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix = self.ix if self.ix is not None else ix, **shape_kwargs)

        p = Condition(self.p, cond_trace=self._cond_trace, as_trace=False) if self._cond_trace is not None else self.p

        p_out = _dispatch()(p)(*shared_args, **inf_kwargs, **shared_kwargs)

        f_out = _dispatch()(self.f)(p_out.trace, p_out.output, **inf_kwargs, **shared_kwargs)

        assert (f_out.log_weight == 0).all()
        assert len(set(f_out.trace.keys()).intersection(set(p_out.trace.keys()))) == 0
        assert len({k for k, v in f_out.trace.items() if v.provenance == Provenance.OBSERVED or v.provenance == Provenance.REUSED}) == 0

        log_u2 = f_out.trace.log_joint(**shape_kwargs, nodes={k for k,v in f_out.trace.items() if v.provenance != Provenance.OBSERVED})

        self._out = Out(
            trace=p_out.trace,
            log_weight=p_out.log_weight + log_u2, # $w_1 \cdot u_2$
            output=p_out.output,
            extras=dict(
                p_out=p_out,
                f_out=f_out,
                trace_star = f_out.trace,
                type=type(self).__name__,
                ix=ix,
                ))

        self._out['loss'] = self.foldr_loss(self._out, maybe(p_out, 'loss', self.loss0))

        return self._out

class Compose(Inf):
    def __init__(
            self,
            q2: Program, # FIXME: make this more general later
            q1: Union[Program, Condition, Resample, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False,
    ) -> None:
        super().__init__(loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.q1 = q1
        self.q2 = q2

    def __call__(self, *shared_args:Any, sample_dims=None, batch_dim=None, _debug=False, _debug_extras=None, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Compose """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix=self.ix if self.ix is not None else ix, **shape_kwargs)

        q1_out = _dispatch()(self.q1)(*shared_args, **inf_kwargs, **shared_kwargs)

        q2_out = _dispatch()(self.q2)(q1_out.trace, q1_out.output, **inf_kwargs, **shared_kwargs)

        assert len(set(q2_out.trace.keys()).intersection(set(q1_out.trace.keys()))) == 0

        self._out = Out(
            trace=trace_utils.copytraces(q2_out.trace, q1_out.trace),
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
            extras=dict(
                q1_out=q1_out,
                q2_out=q2_out,
                type=type(self).__name__,
                ix=ix,
                ))

        self._out['loss'] = self.foldr_loss(self._out, maybe(q1_out, 'loss', self.loss0))

        return self._out


class Propose(Conditionable, Inf):
    def __init__(self,
            p: Union[Program, Extend],
            q: Union[Program, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug:bool=False):
        Conditionable.__init__(self)
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.p = p
        self.q = q

    def __call__(self, *shared_args, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Propose """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix = self.ix if self.ix is not None else ix, **shape_kwargs)

        q_out = _dispatch()(self.q)(*shared_args, **inf_kwargs, **shared_kwargs)

        p_condition = Condition(self.p, q_out.trace)

        p_out = _dispatch()(p_condition)(*shared_args, **inf_kwargs,  **shared_kwargs)

        nodes = set(q_out.trace.keys()) - (
            set({k for k, v in q_out.trace.items() if v.provenance != Provenance.OBSERVED}) \
            - set({k for k, v in p_out.trace.items() if v.provenance != Provenance.OBSERVED})
        )
        u1 = q_out.trace.log_joint(nodes=nodes, **shape_kwargs)

        # τ*, by definition, can't have OBSERVE or REUSED random variables
        u1_star = 0 if 'trace_star' not in q_out else q_out.trace_star.log_joint(**shape_kwargs)

        lw1 = q_out.log_weight
        lv2 = p_out.log_weight - (u1 - u1_star)

        self._out = Out(
            trace=p_out.trace,
            log_weight=lw1 + lv2,
            output=p_out.output,
            extras=dict(
                lv=lv2,
                q_out=q_out,
                p_out=p_out,
                type=type(self).__name__,
                ix=ix,
                ),
            )

        self._out['loss'] = self.foldr_loss(self._out, maybe(q_out, 'loss', self.loss0))

        return self._out
