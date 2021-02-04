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
import combinators.trace.utils as trace_utils

from operator import itemgetter, not_
from combinators.stochastic import Trace, Factor
from combinators.types import Output, check_passable_kwarg, check_passable_arg, Out
from combinators.program import Program, model
from combinators.traceable import TraceModule


class Kernel(TraceModule):
    """ superclass of a program? """
    def __init__(self) -> None:
        super().__init__()

    # TODO: do we need *args? I am thinking no for the time being
    @abstractmethod
    def apply_kernel(self, trace: Trace, cond_trace: Trace, outs: Output, sample_dims:Optional[int]=None, batch_dims=None, **kwargs:Any): #, batch_dim:Optional[int]=None) -> Output:
        raise NotImplementedError()

    def forward(self, cond_trace: Trace, cond_outs: Output, sample_dims:int=None, batch_dims:int=None, validate:bool=True, _debug=False, **kwargs) -> Out:
        types = dict(
            cond_trace=(type(cond_trace), Trace, isinstance(cond_trace, Trace)),
            sample_dims=(type(sample_dims), (int, type(None)), isinstance(sample_dims, (int, type(None)))),
            batch_dims=(type(batch_dims), (int, type(None)), isinstance(batch_dims, (int, type(None)))),
            validate=(type(validate), bool, isinstance(validate, bool)),
        )
        invalid_types = dict(filter(lambda x: not x[1][2], types.items()))

        if len(invalid_types) > 0:
            raise TypeError("\n    ".join([
                "Expected kernel arguments do not typecheck. Current suggestion is to pass auxillary apply_kernel arguments as kwargs.",
                *[f"arg: {k} :: {v[0]}, expected type: {v[1]}" for k, v in invalid_types.items()]
            ]))

        # get a fresh trace to make sure we don't have inplace mutation
        trace = Trace()

        check_kwargs = dict(sample_dims=sample_dims, batch_dims=batch_dims, **kwargs)
        passable_kwargs = {k: v for k,v in check_kwargs.items() if check_passable_kwarg(k, self.apply_kernel)}

        out = self.apply_kernel(trace, cond_trace, cond_outs, **passable_kwargs)

        # grab anything that is missing from the cond_trace
        full_trace = trace_utils.copytraces(cond_trace, trace)

        return Out(full_trace, None, out, extras=dict(type=type(self).__name__))
