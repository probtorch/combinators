#!/usr/bin/env python3

import torch
from torch import nn, Tensor
from typing import Any
from abc import abstractmethod

from combinators.stochastic import Trace, Provenance
from combinators.types import Out, check_passable_kwarg

from combinators.traceable import Conditionable


class Program(nn.Module, Conditionable):
    """ superclass of a program? """
    def __init__(self):
        nn.Module.__init__(self)
        Conditionable.__init__(self)

    @abstractmethod
    def model(self, trace: Trace, *args:Any, **kwargs:Any) -> Any:
        raise NotImplementedError()

    def forward(self, *args:Any, sample_dims=None, batch_dim=None, reparameterized=True, **kwargs:Any):
        shape_kwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim, reparameterized=reparameterized)
        all_kwargs = {**shape_kwargs, **kwargs}

        trace = Trace(cond_trace=self._cond_trace)

        out = self.model(trace, *args, **{k: v for k, v in all_kwargs.items() if check_passable_kwarg(k, self.model)})

        if self._cond_trace is not None:
            # FIXME: move this to Condition
            rho_addrs = {k for k in trace.keys()}
            tau_addrs = {k for k, rv in trace.items() if rv.provenance != Provenance.OBSERVED}
            tau_prime_addrs = {k for k, rv in self._cond_trace.items() if rv.provenance != Provenance.OBSERVED}

            log_weight = trace.log_joint(nodes=rho_addrs - (tau_addrs - tau_prime_addrs), **shape_kwargs)

            self.clear_cond_trace()   # closing bracket for a run, required if a user does not use the Condition combinator
        else:
            log_weight = trace.log_joint(nodes={k for k, rv in trace.items() if rv.provenance == Provenance.OBSERVED}, **shape_kwargs)
            if not isinstance(log_weight, Tensor):
                log_weight = torch.tensor(log_weight)

        return Out(trace=trace, log_weight=log_weight, output=out, extras=dict(type=type(self).__name__, pytype=type(self)))


def dispatch(fn):
    def runit(*args:Any, **kwargs:Any):
        _dispatch_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn)}
        _dispatch_args   = args

        if isinstance(fn, nn.Module):
            _extra_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.forward) and k not in _dispatch_kwargs}
            _dispatch_kwargs = {**_extra_kwargs, **_dispatch_kwargs}

        if isinstance(fn, Program):
            _extra_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.model) and k not in _dispatch_kwargs}
            _dispatch_kwargs = {**_extra_kwargs, **_dispatch_kwargs}

        return fn(*_dispatch_args, **_dispatch_kwargs)
    return runit
