#!/usr/bin/env python3

import torch.nn as nn
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

        trace = self.get_trace()  # allows Condition to hook into this process
        out = self.model(trace, *args, **{k: v for k, v in all_kwargs.items() if check_passable_kwarg(k, self.model)})
        self.clear_cond_trace()   # closing bracket for a run, required if a user does not use the Condition combinator

        log_weight = trace.log_joint(nodes={k for k, rv in trace.items() if rv.provenance == Provenance.OBSERVED}, **shape_kwargs)

        return Out(trace=trace, log_weight=log_weight, output=out, extras=dict(type=type(self).__name__))

