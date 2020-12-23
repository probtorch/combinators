#!/usr/bin/env python3

import torch
import torch.nn as nn
from probtorch import Trace, Factor
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref

from combinators.types import Output, State, TraceLike
from combinators.utils import assert_valid_subtrace, copytrace



@typechecked
class Program(ABC, nn.Module):
    """ superclass of a program? """
    def __init__(self):
        super().__init__()
        self.trace: Optional[Trace] = None

    # FIXME: currently this can only be run once. Need to do some metaprogramming...
    @abstractmethod
    def model(self, trace: Trace, *args:Any) -> Output:
        raise NotImplementedError()

    def log_probs(self, values:Optional[TraceLike] = None) -> Dict[str, Tensor]:
        def eval_under(getter, k):
            is_given = values is not None and k in values.keys()
            log_prob = self.trace[k].dist.log_prob(getter(values, k)) if is_given else self.trace[k].log_prob
            return log_prob

        if values is None:
            return {k: self.trace[k].log_prob for k in self.variables}

        elif isinstance(values, dict):
            return {k: eval_under(lambda vals, k: vals[k], k) for k in self.variables}

        elif isinstance(values, Trace):
            return {k: eval_under(lambda vals, k: vals[k].value, k) for k in self.variables}

    def forward(self, *args:Any, trace: Optional[Trace] = None) -> Tuple[Trace, Output]:
        # Create a new trace every time you run the model forward. not sure if the argument trce is going to cause problems
        self.trace = Trace() if trace is None else trace

        # trace_out, trace_out = self.log_probs(*args) if trace is None else trace
        out = self.model(self.trace, *args)
        # TODO: enforce purity?
        return self.trace, out

    def sample(self, dist:Any, value:Tensor, name:Optional[str]=None) -> None:
        self.trace.append(dist)

        return set(self.trace.keys())

    def observe(self, key:str, value:Tensor) -> None:
        assert key in self.variables, "attempting to abserve a non-existant variable '{}'. Trace contains: {}".format(
            key, ", ".join(self.variables)
        )
        self.trace[key].value = value

        return set(self.trace.keys())

    @property
    def variables(self) -> Set[str]:
        return set(self.trace.keys())

    @classmethod
    def factory(cls, fn, name:str = ""):
        raise RuntimeError('this is broken, clean up OO work first')

        def generic_sample(self, *args, **kwargs):
            return fn(*args, **kwargs)
        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(sample=generic_sample)
        )
        return AProgram()

PROGRAM_REGISTRY = dict()

# FIXME: check in with annotations at a later point
def model(name:str = ""):
    raise RuntimeError('this is broken, clean up OO work first')
    def wrapper(fn):
        global PROGRAM_REGISTRY
        model_key = name + repr(fn)
        instance = Program.factory(fn, name)
        if model_key not in PROGRAM_REGISTRY:
            PROGRAM_REGISTRY[model_key] = instance
        return PROGRAM_REGISTRY[model_key]
    return wrapper

