""" Program introduction forms: OO """

import torch
import torch.nn as nn
from pytest import mark
from torch import Tensor
import hypothesis.strategies as st
from hypothesis import given
from torch.distributions import Normal

from combinators import Program
from combinators.stochastic import equiv, Trace


def test_simple_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()

        def model(self, trace):
            x = trace.normal(loc=0, scale=1, name="x")
            return x

    program = P()
    tr, _, x = program()
    assert isinstance(x, Tensor)
    assert isinstance(tr, Trace)
    assert 'x' in tr

@mark.skip("traces are no longer cached for Observable and, therefore, all log_probs are invalid")
def test_simple_program_log_probs():
    class P(Program):
        def __init__(self):
            super().__init__()

        def model(self, trace):
            x = trace.normal(loc=0, scale=1, name="x")
            return x

    program = P()
    tr, _, x = program()
    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)

    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])

def test_slightly_more_complex_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def model(self, trace, x:int):
            z = trace.normal(loc=0, scale=1, name="z")
            _ = trace.normal(loc=z, scale=1, name="a")
            trace.normal(loc=self.g(z), scale=1, value=torch.tensor(x), name="x")
            return (x, z)

    program = P()
    tr, _, (x, z) = program(1)

    assert isinstance(x, int)
    assert isinstance(z, Tensor)
    assert isinstance(tr, Trace)

    tr2, _, (x2, z2) = program(2)
    assert tr2 is not tr
    assert tr2['x'].value == torch.tensor(2)
    assert not equiv(tr['x'].dist, tr2['x'].dist)

@mark.skip("traces are no longer cached for Observable and, therefore, all log_probs are invalid")
def test_slightly_more_complex_program_logprobs():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def model(self, trace, x:int):
            z = trace.normal(loc=0, scale=1, name="z")
            _ = trace.normal(loc=z, scale=1, name="a")
            trace.normal(loc=self.g(z), scale=1, value=torch.tensor(x), name="x")
            return (x, z)

    program = P()
    tr, _, (x, z) = program(1)
    log_probs = program.log_probs()
    assert set(['x', 'a', 'z']) == program.variables \
        and all([isinstance(log_probs[k], Tensor) for k in program.variables])

    eval_log_probs = program.log_probs(tr)

    for k in program.variables:
       assert torch.allclose(log_probs[k], eval_log_probs[k])

def test_nn_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.in_dim, self.out_dim = 20, 2

            self.g = nn.Sequential(
                nn.Linear(self.in_dim,5),
                nn.ReLU(),
                nn.Linear(5, self.out_dim),
                nn.ReLU()
            )

        def model(self, trace, x:Tensor):
            z = trace.normal(loc=torch.zeros([self.in_dim]), scale=torch.ones([self.in_dim]), name="z")
            _ = trace.normal(loc=z, scale=torch.ones([self.in_dim]), name="a")
            trace.normal(loc=self.g(z), scale=torch.ones([self.out_dim]), value=x, name="x")
            return (x, z)

    program = P()
    tr, _, (x, z) = program(torch.ones([2]))

    assert isinstance(x, Tensor)
    assert isinstance(z, Tensor)
    assert isinstance(tr, Trace)

    # log_probs = program.log_probs()
    # assert set(['x', 'a', 'z']) == program.variables \
    #     and all([isinstance(log_probs[k], Tensor) for k in program.variables])
    #
    # eval_log_probs = program.log_probs(tr)
    #
    # for k in program.variables:
    #    assert torch.allclose(log_probs[k], eval_log_probs[k])


@mark.skip("lazy observes are no longer supported")
def test_generative_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()
            self.g = lambda x: x

        def model(self, trace):
            z = trace.normal(loc=0, scale=1, name="z")
            _ = trace.normal(loc=z, scale=1, name="a")
            x = trace.normal(loc=self.g(z), scale=1, name="observed")
            return (x, z)

    program = P()
    # FIXME: at this point there is nothing on the trace. Alternatively, we could define a `self.model` because here we
    # can only run a sample once -- otherwise we are trying to add new things to a trace a second time
    ones = torch.ones([1])
    program.observe("observed", ones)
    tr, _, (x, z) = program()

    assert torch.allclose(x, ones)
    assert isinstance(z, torch.Tensor)
    # log_probs = program.log_probs()
    # assert 'observed' in log_probs
    # observed_loc = program._trace['observed'].dist.loc
    # assert (log_probs['observed'] == Normal(loc=observed_loc, scale=1).log_prob(ones)).item()



def test_sub_program():
    """
    A couple of ways this can be done but for now here is a naive version
    NOTE: could also do `program.observe("sub.x", torch.ones([1]))`
    """
    class Sub(Program):
        def __init__(self):
            super().__init__()

        def model(self, trace, sub_arg:int):
            x = trace.normal(sub_arg, 1, name="x")
            return x

    class P(Program):
        def __init__(self):
            super().__init__()
            self.sub = Sub()

        def model(self, trace, sub_arg:int, prg_arg:float):
            tr, _, x = self.sub(sub_arg)
            return x * prg_arg

    affine = P()
    mean, scale = 5, 1.3
    tr, _, x = affine(mean, scale)

    assert torch.allclose(x, torch.ones([1]) * mean * scale, rtol=2.5)
