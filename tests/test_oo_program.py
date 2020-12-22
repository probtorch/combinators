""" Program introduction forms: OO """

import torch
from pytest import mark
from probtorch import Trace
from torch import Tensor
import hypothesis.strategies as st
from hypothesis import given

from combinators import Program


def test_simple_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()

        def sample(self, trace):
            x = trace.normal(loc=0, scale=1, name="x")
            return trace, x

    program = P()
    tr, x = program()
    assert isinstance(x, Tensor)
    assert isinstance(tr, Trace)
    assert 'x' in tr
    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)
    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])


def test_slightly_more_complex_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def sample(self, trace, x:int):
            z = self.trace.normal(loc=0, scale=1, name="z")
            _ = self.trace.normal(loc=z, scale=1, name="a")
            self.trace.normal(loc=self.g(z), scale=1, value=torch.tensor(x), name="x")
            return trace, (x, z)

    program = P()
    tr, (x, z) = program(1)

    assert isinstance(x, int)
    assert isinstance(z, Tensor)
    assert isinstance(tr, Trace)
    log_probs = program.log_probs()
    assert set(['x', 'a', 'z']) == program.variables() \
        and all([isinstance(log_probs[k], Tensor) for k in program.variables()])

    eval_log_probs = program.log_probs(tr)

    for k in program.variables():
       assert torch.allclose(log_probs[k], eval_log_probs[k])


@mark.skip("need to do some inspection here or rethink how to define a model")
def test_generative_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def sample(self, trace):
            z = self.trace.normal(loc=0, scale=1, name="z")
            _ = self.trace.normal(loc=z, scale=1, name="a")
            x = self.trace.normal(loc=self.g(z), scale=1, name="x")
            return trace, (x, z)

    program = P()
    # FIXME: at this point there is nothing on the trace. Alternatively, we could define a `self.model` because here we
    # can only run a sample once -- otherwise we are trying to add new things to a trace a second time
    program.observe("x", torch.ones([1]))
    x, z = program()

    assert torch.allclose(x, x)
    assert isinstance(z, torch.Tensor)
    assert program.trace['x'].log_probs is not None
    assert program.trace['z'].log_probs is not None

@mark.skip("this is currently broken. Probably need to define a 'model' instead of 'sample' or do different abstraction")
def test_show_counterexample():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def sample(self, trace):
            z = self.trace.normal(loc=0, scale=1, name="z")
            _ = self.trace.normal(loc=z, scale=1, name="a")
            x = self.trace.normal(loc=self.g(z), scale=1, name="x")
            return trace, (x, z)

    program = P()
    # FIXME: at this point there is nothing on the trace. Alternatively, we could define a `self.model` because here we
    # can only run a sample once -- otherwise we are trying to add new things to a trace a second time
    x, z = program()
    x, z = program()


@mark.skip()
def test_sub_program_creation_option_1():
    """
    A couple of ways this can be done but for now here is a naive version
    NOTE: could also do `program.observe("sub.x", torch.ones([1]))`
    """
    class Sub(Program):
        def __init__(self):
            super().__init__()

        def sample(self, trace):
            x = self.trace.normal(0, 1, name="x")
            return x

    class P(Program):
        def __init__(self):
            self.sub = Sub()

        def sample(self, trace):
            return self.sub()

    program = P()
    program.sub.observe("x", torch.ones([1]))
    x = program()

    assert torch.allclose(x, torch.ones([1]))

@mark.skip()
def test_sub_program_creation_option_2():
    """
    A couple of ways this can be done but for now here is a naive version
    NOTE: could also do `program.observe("sub.x", torch.ones([1]))`
    """
    class Sub(Program):
        def __init__(self):
            super().__init__()

        def sample(self, trace):
            x = self.trace.normal(0, 1, name="x")
            return x

    class P(Program):
        def __init__(self):
            self.sub = Sub()

        def sample(self, trace):
            return self.sub()

    program = P()
    program.observe("sub.x", torch.ones([1]))
    x = program()

    assert torch.allclose(x, torch.ones([1]))
