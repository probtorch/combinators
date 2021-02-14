#!/usr/bin/env python3
from combinators.stochastic import Trace
from collections import ChainMap
from pytest import mark

def test_rerunning_trace():
    t = Trace(idempotent=True)
    x0 = t.normal(0, 1, name="x")
    x1 = t.normal(0, 1, name="x")
    assert x0 is x1
