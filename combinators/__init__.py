#!/usr/bin/env python3

import torch
import torch.nn as nn
from probtorch import Trace, Factor
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod


from combinators.utils import assert_valid_subtrace, copytrace

State   = Any
Samples = Union[Any, List[Any]]
TraceLike = Union[Trace, Dict[str, Tensor]]

@typechecked
class Program(ABC):  # TODO: (nn.Module):
    """ superclass of a program? """
    def __init__(self):
        self.trace = Trace()

    @abstractmethod
    def sample(self, trace: Trace, *args:Any) -> Tuple[Trace, Samples]:
        raise NotImplementedError()

    def _sample(self, trace: Optional[Trace], *args:Any) -> Tuple[Trace, Samples]:
        return self.sample(self.trace if trace is None else trace, *args)

    def log_probs(self, values:Optional[TraceLike] = None) -> Dict[str, Tensor]:
        def eval_under(getter, k):
            is_given = values is not None and k in values.keys()
            log_prob = self.trace[k].dist.log_prob(getter(values, k)) if is_given else self.trace[k].log_prob
            return log_prob

        if values is None:
            return {k: self.trace[k].log_prob for k in self.variables()}

        elif isinstance(values, dict):
            return {k: eval_under(lambda vals, k: vals[k], k) for k in self.variables()}

        elif isinstance(values, Trace):
            return {k: eval_under(lambda vals, k: vals[k].value, k) for k in self.variables()}

    def __call__(self, *args:Any, trace: Optional[Trace] = None) -> Tuple[Trace, Samples]:
        tr = self.trace if trace is None else trace
        # trace_out, trace_out = self.log_probs(*args) if trace is None else trace
        trace_out, out = self._sample(tr, *args)
        # TODO: enforce purity?
        return trace_out, out

    def variables(self) -> Set[str]:
        return set(self.trace.keys())

    def observe(self, key:str, value:Tensor) -> None:
        assert key in self.variables(), "attempting to abserve a non-existant variable '{}'. Trace contains: {}".format(
            key, ", ".join(self.variables())
        )
        self.trace[key].value = value

        return set(self.trace.keys())

    @classmethod
    def factory(cls, fn, name:str = ""):
        def generic_sample(self, *args, **kwargs):
            return fn(*args, **kwargs)

        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(sample=generic_sample)
        )
        return AProgram()


# FIXME: currently not used, but currying the annotations might be nice
def curry(func):
    """ taken from: https://www.python-course.eu/currying_in_python.php """
    curry.__curried_func_name__ = func.__name__
    f_args, f_kwargs = [], {}
    def f(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            return f
        else:
            result = func(*f_args, *f_kwargs)
            f_args, f_kwargs = [], {}
            return result
    return f

PROGRAM_REGISTRY = dict()

def model(name:str = ""):
    def wrapper(fn):
        global PROGRAM_REGISTRY
        model_key = name + repr(fn)
        instance = Program.factory(fn, name)
        if model_key not in PROGRAM_REGISTRY:
            PROGRAM_REGISTRY[model_key] = instance
        return PROGRAM_REGISTRY[model_key]
    return wrapper


class Kernel:
    def __init__(self):
        self.trace = Trace()

    def evaluate(self, state: State, cond_trace: Trace, obs: Samples) -> Tuple[State, Trace]:
        """ evaluate observations under given trace, return logprobs """
        # automatically implied under the hood
        raise NotImplementedError()

    def _evaluate(self, state: State, cond_trace: Trace, obs: Samples) -> Tuple[State, Trace, Samples]:
        """ evaluate the samples under the conditioned trace, returning the trace """
        state_out, trace_out = self.evaluate(state, cond_trace, obs)
        trace_out = trace_out + copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys())))
        return state_out, trace_out, obs

    def generate(self, state: State, cond_trace: Trace) -> Tuple[State, Samples]:
        """ generate logprobs, run them through kernel with program tr """
        raise NotImplementedError()

    def _generate(self, state: State, cond_trace: Trace) -> Tuple[State, Trace, Samples]:
        """ produce samples for a RV that I produce, returning the samples and trace """
        state_out,       out = self.generate(state, cond_trace)
        state_out, trace_out = self.evaluate(state, cond_trace, out)
        trace_out = trace_out + copytrace(cond_trace, set(cond_trace.keys() - set(trace_out.keys())))
        return state_out, trace_out, out

    def __call__(self, state: State, cond_trace: Trace, x: Samples = None) -> Tuple[State, Trace, Samples]:
        # tr_{prg} <: tr_{ker} -- ie: all of kernel tr in prg tr
        assert_valid_subtrace(cond_trace, self.trace)

        if x is None:
            return self._generate(state, cond_trace)
        else:
            return self._evaluate(state, cond_trace, x)


class Reverse:
    def __init__(self, proposal: Program, kernel: Kernel):
        self.proposal = proposal
        self.kernel = kernel

    def __call__(self, state: State) -> Tuple[State, Trace, Samples]:
        return self.kernel._generate(state, self.proposal.trace)


class Forward:
    def __init__(self, kernel: Kernel, target: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Samples) -> Tuple[State, Trace, Samples]:
        return self.kernel._evaluate(state, self.proposal.trace, x)


class Propose:
    def __init__(self, target: Program, proposal: Program):
        self.target = target
        self.kernel = kernel

    def __call__(self, state: State, x: Samples) -> Tuple[State, Trace, Samples]:
        return self.kernel._evaluate(state, self.proposal.trace, x)
