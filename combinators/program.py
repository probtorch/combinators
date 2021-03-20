import torch
import inspect
from torch import nn, Tensor
from typing import Any, Optional
from abc import abstractmethod, ABC

from combinators.stochastic import Trace, Provenance
from combinators.out import Out

# FIXME: move to probtorch
def copytraces(*traces, exclude_nodes=None):
    """
    merge traces together. domains should be disjoint otherwise last-write-wins.
    """
    newtrace = Trace()
    if exclude_nodes is None:
        exclude_nodes = {}

    for tr in traces:
        for k, rv in tr.items():
            if k in exclude_nodes:
                continue
            newtrace.append(rv, name=k)
    return newtrace

class Conditionable(ABC):
    def __init__(self) -> None:
        super().__init__()
        # Optional is used to tell us if a program has a conditioning trace, as opposed to
        # a misspecified condition on the empty trace
        self._cond_trace: Optional[Trace] = None


class EvalSubCtx(object):
    def __init__(self, program: Conditionable, subtitution_trace: Optional[Trace]=None) -> None:
        self.program = program
        # FIXME: do we actually need a copy of the trace? Might cause a dangling ref.
        self.subtitution_trace = None if subtitution_trace is None else copytraces(subtitution_trace)

    def __enter__(self):
        self.program._cond_trace = self.subtitution_trace

    def __exit__(self, type, value, traceback):
        self.program._cond_trace = None



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

        # when we thread the inference state through all of the combinators, we should have
        # the evalsubctx handle this.
        trace = Trace(cond_trace=self._cond_trace)
        out = self.model(trace, *args, **{k: v for k, v in all_kwargs.items() if check_passable_kwarg(k, self.model)})

        #if self._cond_trace is not None:
        # TODO: move this to inference.Condition
        rho_addrs = {k for k in trace.keys()}
        tau_addrs = {k for k, rv in trace.items() if rv.provenance != Provenance.OBSERVED}
        tau_prime_addrs = {k for k, rv in self._cond_trace.items() if rv.provenance != Provenance.OBSERVED} \
            if self._cond_trace is not None else set()

        log_weight = trace.log_joint(nodes=rho_addrs - (tau_addrs - tau_prime_addrs), **shape_kwargs)

        trace._cond_trace = None
        #self._cond_trace = None   # required if a user does not use the Condition combinator
        #else:
        #    log_weight = trace.log_joint(nodes={k for k, rv in trace.items() if rv.provenance == Provenance.OBSERVED}, **shape_kwargs)

        return Out(trace=trace, log_weight=log_weight, output=out, extras=dict(type=type(self).__name__, pytype=type(self)))


def check_passable_kwarg(name, fn):
    ''' check if a kwarg can be passed into a function '''
    fullspec = inspect.getfullargspec(fn)
    return fullspec.varkw is not None or name in fullspec.kwonlyargs or name in fullspec.args


def dispatch(fn):
    ''' given a function, pass all *args and any **kwargs that type-check '''

    def runit(*args:Any, **kwargs:Any):
        # TODO: We can reduce this to only one branch
        _dispatch_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn)}
        _dispatch_args   = args

        if isinstance(fn, nn.Module):
            ''' additionally, if the function is an nn.Module, we need to check forward '''
            _extra_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.forward) and k not in _dispatch_kwargs}
            _dispatch_kwargs = {**_extra_kwargs, **_dispatch_kwargs}

        if isinstance(fn, Program):
            ''' additionally, if the function is a Program, we need to check model '''
            _extra_kwargs = {k: v for k,v in kwargs.items() if check_passable_kwarg(k, fn.model) and k not in _dispatch_kwargs}
            _dispatch_kwargs = {**_extra_kwargs, **_dispatch_kwargs}

        return fn(*_dispatch_args, **_dispatch_kwargs)
    return runit
