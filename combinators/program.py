#!/usr/bin/env python3

import torch
import torch.nn as nn
from probtorch import Trace, Factor
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod

from combinators.types import Samples, State, TraceLike
from combinators.utils import assert_valid_subtrace, copytrace


@typechecked
class Program(ABC):  # TODO: (nn.Module):
    """ superclass of a program? """
    def __init__(self):
        self.trace = Trace()


    # FIXME: currently this can only be run once. Need to do some more metaprogramming...
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

    # TODO: verify that this works
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

