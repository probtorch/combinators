#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Any, Tuple, Optional, Union, Set, Callable
from abc import ABC
from typing import NamedTuple

import combinators.debug as debug
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
import combinators.resampling.strategies as rstrat

from combinators.types import check_passable_kwarg, Out
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.stochastic import Trace, Provenance, RandomVariable, ImproperRandomVariable
from combinators.program import Program, dispatch
from combinators.kernel import Kernel
from combinators.traceable import Conditionable
from combinators.metrics import effective_sample_size

def copytraces(*traces, exclude_node=None):
    newtrace = Trace()
    if exclude_node is None:
        exclude_node = {}

    for tr in traces:
        for k, rv in tr.items():
            if k in exclude_node:
                break
            newtrace.append(rv, name=k)
    return newtrace

def rerun_with_detached_values(trace:Trace):
    newtrace = Trace()
    def rerun_rv(rv):
        value = rv.value.detach()
        if isinstance(rv, RandomVariable):
            return RandomVariable(value=value, dist=rv.dist, provenance=rv.provenance, reparameterized=rv.reparameterized)
        elif isinstance(rv, ImproperRandomVariable):
            return ImproperRandomVariable(value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance)
        else:
            raise NotImplementedError("Only supports RandomVariable and ImproperRandomVariable")
    for k, v in trace.items():
        newtrace.append(rerun_rv(v), name=k)
    return newtrace

def maybe(obj, name, default, fn=(lambda x: x)):
    return fn(getattr(obj, name)) if hasattr(obj, name) else default


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
        self.conditioning_trace = copytraces(cond_trace)

        self._requires_grad = requires_grad
        self._detach = detach
        self.as_trace = as_trace
        self.full_trace_return = full_trace_return

    def __call__(self, c:Any, _debug=False, **kwargs:Any) -> Out:
        """ Condition """

        self.program._cond_trace = self.conditioning_trace

        out = dispatch(self.program)(c, **kwargs)

        out['type']=type(self).__name__ + "(" + type(self.program).__name__ + ")"
        # out['cond_trace']=self.program._cond_trace

        self.program._cond_trace = None
        # Also clear cond_trace reference in trace to not introduce a space leak
        out.trace._cond_trace = None

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
            strategy=None
    ):
        Inf.__init__(self, ix=ix, _debug=_debug, loss0=loss0)
        self.q = q
        self.strategy = rstrat.Systematic()

    def __call__(self, c, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Resample """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix=self.ix if self.ix is not None else ix, **shape_kwargs)

        q_out = self.q(c, **inf_kwargs, **shared_kwargs)

        passable_kwargs = {k: v for k, v in shape_kwargs.items() if check_passable_kwarg(k, self.strategy)}

        tr_2, lw_2 = self.strategy(q_out.trace, q_out.log_weight, **passable_kwargs)

        c1 = q_out.output
        assert isinstance(c1, dict)
        c2 = {k: v for k, v in c1.items()}
        rs_out_addrs = set(c1.keys()).intersection(set(tr_2.keys()))
        for rs_out_addr in rs_out_addrs:
            assert isinstance(c2[rs_out_addr], torch.Tensor)
            c2[rs_out_addr] = tr_2[rs_out_addr].value

        self._out = Out(
            extras=dict(
                q_out=q_out,
                type=type(self).__name__,
                ix=ix,
                ),
            trace=tr_2,
            log_weight=lw_2,
            output=c2,
        )

        self._out['loss'] = self.foldr_loss(self._out, maybe(q_out, 'loss', self.loss0))

        return self._out


class Extend(Inf, Conditionable):
    def __init__(self,
            p: Program, # FIXME: make this :=  p | extend (p, f) later
            f: Program,
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug=False) -> None:
        Conditionable.__init__(self)
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.p = p
        self.f = f

    def __call__(self, c:Any, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs:Any) -> Out:
        """ Extend """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix = self.ix if self.ix is not None else ix, **shape_kwargs)

        if self._cond_trace is None:
            p_out = dispatch(self.p)(c, **inf_kwargs, **shared_kwargs)

            f_out = dispatch(self.f)(p_out.output, **inf_kwargs, **shared_kwargs)

            assert (f_out.log_weight == 0.0)
            assert len({k for k, v in f_out.trace.items() if v.provenance == Provenance.OBSERVED or v.provenance == Provenance.REUSED}) == 0

        else:
            p_out = dispatch(Condition(self.p, self._cond_trace))(c, **inf_kwargs, **shared_kwargs)

            f_out = dispatch(Condition(self.f, self._cond_trace))(p_out.output, **inf_kwargs, **shared_kwargs)

            assert len({k for k, v in f_out.trace.items() if v.provenance == Provenance.OBSERVED}) == 0

        assert len(set(f_out.trace.keys()).intersection(set(p_out.trace.keys()))) == 0

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
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        self.q1 = q1
        self.q2 = q2

    def __call__(self, c:Any, sample_dims=None, batch_dim=None, _debug=False, _debug_extras=None, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Compose """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)

        inf_kwargs = dict(_debug=_debug, ix=self.ix if self.ix is not None else ix, **shape_kwargs)

        q1_out = dispatch(self.q1)(c, **inf_kwargs, **shared_kwargs)

        q2_out = dispatch(self.q2)(q1_out.output, **inf_kwargs, **shared_kwargs)

        assert len(set(q2_out.trace.keys()).intersection(set(q1_out.trace.keys()))) == 0, "addresses must not overlap"

        self._out = Out(
            trace=copytraces(q2_out.trace, q1_out.trace),
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


class Propose(Inf):
    def __init__(self,
            p: Union[Program, Extend],
            q: Union[Program, Inf],
            loss_fn=(lambda x, fin: fin),
            loss0=None,
            device=None,
            ix=None,
            _debug:bool=False):
        Inf.__init__(self, loss_fn=loss_fn, loss0=loss0, device=device, ix=ix, _debug=_debug)
        assert not isinstance(p, Compose)
        self.p = p
        self.q = q

    def __call__(self, c, sample_dims=None, batch_dim=None, _debug=False, reparameterized=True, ix=None, **shared_kwargs) -> Out:
        """ Propose """
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        inf_kwargs = dict(_debug=_debug, ix = self.ix if self.ix is not None else ix, **shape_kwargs)

        q_out = dispatch(self.q)(c, **inf_kwargs, **shared_kwargs)

        p_condition = Condition(self.p, q_out.trace)

        p_out = dispatch(p_condition)(c, **inf_kwargs,  **shared_kwargs)

        rho_1 = set(q_out.trace.keys())
        tau_1 = set({k for k, v in q_out.trace.items() if v.provenance != Provenance.OBSERVED})
        tau_2 = set({k for k, v in p_out.trace.items() if v.provenance != Provenance.OBSERVED})
        nodes = rho_1 - (tau_1 - tau_2)
        lu_1 = q_out.trace.log_joint(nodes=nodes, **shape_kwargs)

        # τ*, by definition, can't have OBSERVE or REUSED random variables
        lu_star = torch.zeros(1) if 'trace_star' not in p_out else q_out.trace.log_joint(nodes=set(p_out.trace_star.keys()), **shape_kwargs)

        lw_1 = q_out.log_weight
        # We call that lv because its the incremental weight in the IS sense
        # In the semantics this corresponds to lw_2 - (lu + [lu_star])
        lv = p_out.log_weight - (lu_1 + lu_star)
        lw_out = lw_1 + lv

        self._out = Out(
            trace=rerun_with_detached_values(p_out.trace),
            log_weight=lw_out.detach(),
            output=p_out.output,
            extras=dict(
                # FIXME: Delete before publishing - this is for debugging only
                lu=(lu_1 + lu_star),
                lu_1=lu_1,
                lu_star=lu_star,
                rho_1=rho_1,
                tau_1=tau_1,
                tau_2=tau_2,
                nodes=nodes,
                ## stats ##
                ess = effective_sample_size(lw_out, sample_dims=sample_dims),
                ## apg ##
                p_num=p_out.p_out.log_weight if (p_out.type == "Extend") else p_out.log_weight,
                q_den=lu_star,
                #########
                trace_original=p_out.trace,
                lv=lv,
                q_out=q_out,
                p_out=p_out,
                type=type(self).__name__,
                # FIXME: can we ditch this? how important is this for objectives
                trace_star=p_out.trace_star if 'trace_star' in p_out else None,
                ix=ix,
                ),
        )
        self._out['loss'] = self.foldr_loss(self._out, maybe(q_out, 'loss', self.loss0))
        return self._out
