#!/usr/bin/env python3

import torch
import torch.nn as nn
from pytest import mark
from torch import Tensor
import hypothesis.strategies as st
from hypothesis import given
from torch.distributions import Normal

from combinators import Program, Propose
from combinators.stochastic import equiv, Trace


@mark.skip()
def test_propose():
    """
    A couple of ways this can be done but for now here is a naive version
    NOTE: could also do `program.observe("sub.x", torch.ones([1]))`
    """
    class Subber(Program):
        def __init__(self):
            super().__init__()

    class Sub(Program):
        def __init__(self):
            super().__init__()
            self.subber = Subber()

        def sample(self, trace):
            x = self.trace.normal(0, 1, name="x")
            return x


    class P(Program):
        def __init__(self):
            self.sub = Sub()

        def sample(self, trace):
            return self.sub()

    # propose = Propose(P(), Q())
    #
    # propose().trace = {
    #     'x': ...
    #
    #     'q.x': ...
    #     'p.x': ...
    #
    #     'q.y': ...  # could 'assume that same variable exists in p'
    #     # w = \frac{p(x) } / {q(x)}
    #     # w = \frac{p(x) } / {q(x)q(y)}
    #     # w = \frac{p(x) q(y)} / {q(x)q(y)} <== "assume q(y) in p so that it cancels in the weights"
    #     # w = intersection(q.keys(), p.keys())
    #     # tr = union(q.keys(), p.keys())
    #
    #     'p.z': ...
    # }

    propose = Propose(P(), Q())
    # tr, output = propose(*p_args)(*q_args)
    tr, (p_outputs, q_outs) = propose(*p_args)(*q_args)



    program.observe("sub.x", torch.ones([1]))
    x = program()

    assert torch.allclose(x, torch.ones([1]))
