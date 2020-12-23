""" Program introduction forms: OO """

import torch
import torch.nn as nn
from pytest import mark
from probtorch import Trace
from torch import Tensor
import hypothesis.strategies as st
from hypothesis import given

from combinators import Program
from combinators.stochastic import equiv


def test_simple_program_creation():
    class P(Program):
        def __init__(self):
            super().__init__()

        def model(self, trace):
            x = trace.normal(loc=0, scale=1, name="x")
            return x

    program = P()
    tr, x = program()
    assert isinstance(x, Tensor)
    assert isinstance(tr, Trace)
    assert 'x' in tr
    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)

    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])

# 1d gaussian
#  - pi_1 1d gaus mean 0
#  - pi_2 1d gaus mean 1   <<< at one step no need for detaches in the NVI step (only if you don't compute normalizing constants)
#  - pi_3 1d gaus mean 2
#  - pi_4 1d gaus mean 3
#
# NVI stuff -- target and proposal always fixed
#           -- detaches happen in between (don't forget)
#
# 1-step NVI (VAE)
# 3-step NVI (NVI-sequential): 4 intermediate densities
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
    tr, (x, z) = program(1)

    assert isinstance(x, int)
    assert isinstance(z, Tensor)
    assert isinstance(tr, Trace)
    log_probs = program.log_probs()
    assert set(['x', 'a', 'z']) == program.variables \
        and all([isinstance(log_probs[k], Tensor) for k in program.variables])

    eval_log_probs = program.log_probs(tr)

    for k in program.variables:
       assert torch.allclose(log_probs[k], eval_log_probs[k])

    tr2, (x2, z2) = program(2)
    assert tr2 is not tr
    assert tr2['x'].value == torch.tensor(2)
    assert not equiv(tr['x'].dist, tr2['x'].dist)




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
