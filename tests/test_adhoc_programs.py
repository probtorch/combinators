""" Program introduction forms: Annotations """

import torch
from pytest import mark
from probtorch.stochastic import Trace
from torch import Tensor
import hypothesis.strategies as st
from hypothesis import given

# TODO: annotation to convert pure functions to Programs is sugar for later
_ = '''
from combinators import model


@mark.skip()
def test_simple_program_creation():
    @model()
    def program(trace):
        x = trace.normal(loc=0, scale=1, name="x")
        return x

    tr, x = program()

    assert isinstance(x, Tensor)
    assert isinstance(tr, Trace)
    assert 'x' in tr
    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)
    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])


@mark.skip()
def test_slightly_more_complex_program_creation():
    @model()
    def program(trace, g, x):
        z = trace.normal(loc=0, scale=1, name="z")
        _ = trace.normal(loc=z, scale=1, name="a")
        trace.normal(loc=g(z), scale=1, value=torch.tensor(x), name="x")
        return x, z

    g = lambda x: x
    # FIXME: all of these should work eventually. Need to think of some ways to do fancy currying
    # tr, (x, z) = program()(g, 1)
    # tr, (x, z) = program(g)(1)
    # tr, (x, z) = program(g, 1)
    # tr, (x, z) = program()(g)(1)

    tr, (x, z) = program(g, 1)
    assert isinstance(x, int)
    assert isinstance(z, Tensor)
    assert isinstance(tr, Trace)
    log_probs = program.log_probs()
    assert set(['x', 'a', 'z']) == program.variables \
        and all([isinstance(log_probs[k], Tensor) for k in program.variables])

    eval_log_probs = program.log_probs(tr)

    for k in program.variables:
       assert torch.equal(log_probs[k], eval_log_probs[k])
'''
