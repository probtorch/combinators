#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.utils import assert_valid_subtrace, copytrace

from combinators.traceable import Traceable


@typechecked
class Program(Traceable, nn.Module):
    """ superclass of a program? """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def model(self, trace: Trace, *args:Any) -> Output:
        raise NotImplementedError()

    def forward(self, *args:Any, trace: Optional[Trace] = None) -> Tuple[Trace, Output]:
        # FIXME: Create a new trace every time you run the model forward. not sure if the argument trce is going to cause problems
        trace = self._apply_observes(self.get_trace(evict=True) if trace is None else trace)

        # trace_out, trace_out = self.log_probs(*args) if trace is None else trace
        out = self.model(trace, *args)
        # TODO: enforce purity?
        return trace, out

    @classmethod
    def factory(cls, fn, name:str = ""):
        raise RuntimeError('this is broken, clean up OO work first')

        def generic_sample(self, *args, **kwargs):
            return fn(*args, **kwargs)
        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(sample=generic_sample)
        )
        return AProgram()


class Propose(nn.Module):
    def __init__(self, target: Program, proposal: Program):
        super().__init__()
        self.target = target
        self.proposal = proposal

    def forward(self, *target_args):
        target_trace, _ = self.target(*target_args)

        def run_proposal(*proposal_args):
            return self.propsal(*proposal_args, trace=target_trace)

        return run_proposal

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

