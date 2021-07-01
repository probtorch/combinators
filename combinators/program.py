import torch
import inspect
from torch import nn, Tensor
from typing import Any, Optional
from abc import abstractmethod, ABC

from probtorch.stochastic import _RandomVariable, Trace, Provenance
from combinators.trace.utils import copytraces
from combinators.out import Out


class Conditionable(ABC):
    def __init__(self) -> None:
        super().__init__()
        # Optional is used to tell us if a program has a conditioning trace, as opposed to
        # a misspecified condition on the empty trace
        self._cond_trace: Optional[Trace] = None


class WithSubstitution(object):
    def __init__(
        self, program: Conditionable, subtitution_trace: Optional[Trace] = None
    ) -> None:
        self.program = program
        # FIXME: do we actually need a copy of the trace? Might cause a dangling ref.
        self.subtitution_trace = (
            None if subtitution_trace is None else copytraces(subtitution_trace)
        )

    def __enter__(self):
        self.program._cond_trace = self.subtitution_trace

    def __exit__(self, type, value, traceback):
        self.program._cond_trace = None


class Program(nn.Module, Conditionable):
    """superclass of a program?"""

    def __init__(self):
        nn.Module.__init__(self)
        Conditionable.__init__(self)

    @abstractmethod
    def model(self, trace: Trace, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def forward(
        self,
        *args: Any,
        sample_dims=None,
        batch_dim=None,
        reparameterized=True,
        **kwargs: Any
    ):
        shape_kwargs = dict(
            sample_dims=sample_dims,
            batch_dim=batch_dim,
            reparameterized=reparameterized,
        )
        all_kwargs = {**shape_kwargs, **kwargs}

        # when we thread the inference state through all of the combinators, we should have
        # the SubstitutionCtx handle this.
        trace = Trace(cond_trace=self._cond_trace)
        out = self.model(
            trace,
            *args,
            **{
                k: v
                for k, v in all_kwargs.items()
                if check_passable_kwarg(k, self.model)
            }
        )

        rho_addrs = {k for k in trace.keys()}
        tau_filter = (
            lambda rv: isinstance(rv, _RandomVariable)
            and rv.provenance != Provenance.OBSERVED
        )
        tau_addrs = {k for k, rv in trace.items() if tau_filter(rv)}
        tau_prime_addrs = (
            {k for k, rv in self._cond_trace.items() if tau_filter(rv)}
            if self._cond_trace is not None
            else set()
        )

        log_weight = trace.log_joint(
            nodes=rho_addrs - (tau_addrs - tau_prime_addrs), **shape_kwargs
        )

        # clean up any substitution context
        trace._cond_trace = None

        return Out(
            trace=trace,
            log_weight=log_weight,
            output=out,
            extras=dict(type=type(self).__name__, pytype=type(self)),
        )


def check_passable_kwarg(name, fn):
    """check if a kwarg can be passed into a function"""
    fullspec = inspect.getfullargspec(fn)
    return (
        fullspec.varkw is not None
        or name in fullspec.kwonlyargs
        or name in fullspec.args
    )


def dispatch(fn, *args: Any, **kwargs: Any):
    """given a function, pass all *args and any **kwargs that type-check"""

    def kwargs_for(_f, _nxt, _prev=None):
        _prev = (
            dict() if _prev is None else _prev
        )  # always hedge against sneaky globals
        return {
            k: v
            for k, v in _nxt.items()
            if check_passable_kwarg(k, _f) and k not in _prev
        }

    _dispatch_kwargs = kwargs_for(fn, kwargs)

    if isinstance(fn, nn.Module):
        """additionally, if the function is an nn.Module, we need to check forward"""
        _dispatch_kwargs = {
            **kwargs_for(fn.forward, kwargs, _dispatch_kwargs),
            **_dispatch_kwargs,
        }

    if isinstance(fn, Program):
        """additionally, if the function is a Program, we need to check model"""
        _dispatch_kwargs = {
            **kwargs_for(fn.model, kwargs, _dispatch_kwargs),
            **_dispatch_kwargs,
        }

    return fn(*args, **_dispatch_kwargs)
