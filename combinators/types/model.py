#!/usr/bin/env python3

import torch
from probtorch import Trace
from torch import Tensor

# local imports
from combinators.types.combinator import Combinator, Output, Weight, Callable, Any, Tuple, NewType


State = Any
Program = NewType("Program", Callable[[State], Tuple[State, Output]])



class Program(Combinator):
    """
    A probabilistic program defines an (unnormalized) density over traces.

    We will require that, for any program $f$ with inputs $x$, it is possible to point-wise evaluate an unnormalized
    density and compute the return value from the trace and the inputs

    """

    def __init__(self, trace: Trace):
        self.trace = trace

    def __call__(self, samples: Tensor) -> Output:
        raise NotImplementedError()

    def evaluate(self, samples: Tensor, trace: Trace) -> Tensor:
        raise NotImplementedError()

    def weight(self, samples: Tensor, trace: Trace) -> Weight:
        """ ω_f(τ ; x) """
        raise NotImplementedError()

    def dist(self, samples: Tensor, trace: Trace) -> Tensor:
        """ p_f(x; τ) """
        raise NotImplementedError()

    def __repr__(self) -> str:
        def trace_item(key: str, aten: Tensor) -> str:
            return "{}|->Tensor[{}]".format(key, aten.shape)

        trace_items = [trace_item(key, ten) for key, ten in self.trace.items()]
        return "Model({})".format("; ".join(trace_items))


class Stub(Program):
    def __init__(self) -> None:
        super().__init__(Trace())

    def __call__(self, samples: Tensor) -> Output:
        return samples

    def evaluate(self, samples: Tensor, trace: Trace) -> Trace:
        return trace

    def weight(self, samples: Tensor, trace: Trace) -> Weight:
        return Weight(torch.ones(samples.shape))

    def dist(self, samples: Tensor, trace: Trace) -> Trace:
        return self.trace


stub = Stub()
