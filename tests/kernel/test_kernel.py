""" kernel introduction forms """
import torch
from pytest import mark, raises
from torch import Tensor
from torch.distributions import Normal
import hypothesis.strategies as st
from hypothesis import given

from combinators import Program, Kernel, Propose, Forward
from combinators.stochastic import equiv, Trace


def test_kernel_creation():
    class K(Kernel):
        def __init__(self):
            super().__init__()

        def apply_kernel(self, trace, cond_trace, obs):
            x, y = obs
            z = trace.normal(loc=0, scale=1, name="z")
            return x + z

    kernel = K()
    cond_trace = Trace()
    cond_trace.normal(loc=-1, scale=3, name='z')
    tr, out = kernel(cond_trace, (1, 2))

    with raises(Exception):
        cond_trace = Trace()
        tr, out = kernel(cond_trace, (1, 2))

    with raises(Exception):
        cond_trace = Trace()
        cond_trace.normal(loc=-1, scale=3, name='q')
        tr, out = kernel(cond_trace, (1, 2))

@mark.skip()
def test_forward():
    class P(Program):
        def __init__(self):
            super().__init__()

        def sample(self, trace, y:int, z:int):
            x = trace.normal(loc=0, scale=1, name="x")
            return (x, y)

    class K(Kernel):
        def __init__(self):
            super().__init__()

        def apply(self, trace, cond_trace, obs):
            x, y = obs
            z = trace.normal(loc=0, scale=1, name="z")
            return x + z

    run_program_followed_by_kernel = Forward(K(),P())  # implicitly runs program? at least for now to get moving...

    run_kernel = run_program_followed_by_kernel(1, 2)

    # run_program_followed_by_kernel.program_outputs

    k_tr, x = run_kernel()

    assert isinstance(x, Tensor)
    assert isinstance(k_tr, Trace)

    assert 'y' in k_tr
    assert 'x' in program.trace

    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)
    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])

@mark.skip()
def test_forward():
    class P(Program):
        def __init__(self):
            super().__init__()

        def sample(self, trace, y:int, z:int):
            x = trace.normal(loc=0, scale=1, name="x")
            return (x, y)

    class K(Kernel):
        def __init__(self):
            super().__init__()

        def apply(self, trace, cond_trace, obs):
            x, y = obs
            z = trace.normal(loc=0, scale=1, name="z")
            return x + z

    run_program_followed_by_kernel = Forward(K(),P())  # implicitly runs program? at least for now to get moving...

    run_kernel = run_program_followed_by_kernel(1, 2)

    # run_program_followed_by_kernel.program_outputs

    k_tr, x = run_kernel()

    assert isinstance(x, Tensor)
    assert isinstance(k_tr, Trace)

    assert 'y' in k_tr
    assert 'x' in program.trace

    log_probs = program.log_probs()
    assert 'x' in log_probs.keys() and isinstance(log_probs['x'], Tensor)
    eval_log_probs = program.log_probs(dict(x=torch.ones_like(log_probs['x']) / log_probs['x']))
    assert 'x' in eval_log_probs.keys() and not torch.allclose(log_probs['x'], eval_log_probs['x'])
