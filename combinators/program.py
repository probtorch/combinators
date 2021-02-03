#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
import ast
import weakref

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike, get_shape_kwargs, Out
import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad

from combinators.traceable import TraceModule

class Program(TraceModule):
    """ superclass of a program? """
    def __init__(self, with_joint=None):
        super().__init__()
        self._with_joint=with_joint

    @abstractmethod
    def model(self, trace: Trace, *args:Any, **kwargs:Any) -> Output:
        raise NotImplementedError()

    def forward(self, *args:Any, sample_dims=None, batch_dim=None, reparameterized=True, **kwargs:Any):
        trace = self.get_trace()  # allows Condition to hook into this process
        skwargs = dict(sample_dims=sample_dims, batch_dim=batch_dim)
        out = self.model(trace, *args, **get_shape_kwargs(self.model, **skwargs), **kwargs)


        self.clear_cond_trace()   # closing bracket for a run, required if a user does not use the Condition combinator

        jkwargs = dict(reparameterized=reparameterized, **skwargs)

        # TODO: there is some precision error when using nodes, so:
        log_joint = trace_utils.copysubtrace(trace, self._with_joint).log_joint(**jkwargs) \
            if self._with_joint is not None else trace.log_joint(**jkwargs)

        return Out(trace, log_joint, out, extras=dict(type=type(self).__name__))

    @classmethod
    def factory(cls, fn, name:str = ""):
        def generic_model(self, *args, **kwargs):
            return fn(*args, **kwargs)
            # import ipdb; ipdb.set_trace();
            #
            # if not isinstance(out, (tuple, list)):
            #     raise TypeError("ad-hoc models are expected to return a tuple or list with the input trace as the first return.")
            # elif not isinstance(out[0], Trace):
            #     # just being lazy here
            #     raise TypeError("ad-hoc models are expected to return a tuple or list with the input trace as the first return.")
            # elif len(out) == 1:
            #     # okay not about to think about this part very hard...
            #     return out[0], None
            # elif len(out) == 2:
            #     return out
            # else:
            #     # let users slide here, but this seems pretty painful
            #     trace = out[0]
            #     final = out[1:]
            #     return trace, final

        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(model=generic_model)
        )

        return AProgram()

    def copy(self):
        def generic_model(self, *args, **kwargs):
            return fn(*args, **kwargs)

        AProgram = type(
            "AProgram<{}>".format(repr(fn)), (cls,), dict(model=generic_model)
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

