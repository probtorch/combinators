""" kernel introduction forms """
import torch
import torch.nn as nn
from pytest import mark, raises
from torch import Tensor
from torch.distributions import Normal
import hypothesis.strategies as st
from hypothesis import given

from combinators import Program, Kernel, Forward, Reverse, Trace


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


class Simple(Program):
    in_dim, out_dim = 8, 2
    def __init__(self):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(Simple.in_dim, Simple.in_dim // 2),
            nn.ReLU(),
            nn.Linear(Simple.in_dim // 2, Simple.out_dim),
        )

    def model(self, trace, x:Tensor):
        loc1 = torch.zeros([Simple.in_dim])
        scale1, scale2 = torch.ones([Simple.in_dim]), torch.ones([Simple.out_dim])

        z = trace.normal(loc=loc1, scale=scale1, name="z")
        x_fixed = trace.normal(loc=self.g(z), scale=scale2, value=x, name="x")
        assert x_fixed is x
        return (x, z)

class Kernel(Kernel):
    def __init__(self):
        super().__init__()

    def apply_kernel(self, trace, cond_trace, obs):
        mkten = lambda size, scale: torch.ones([size]) * scale

        x, z = obs
        z_k = trace.normal(loc=mkten(Simple.in_dim,  20), scale=mkten(Simple.in_dim, 1), name="z")
        x_k = trace.normal(loc=mkten(Simple.out_dim, -20), scale=mkten(Simple.out_dim, 1), name="x")
        return (x / x_k, z / z_k)


def test_forward():
    program = Simple()
    prg_inp = torch.ones([program.out_dim])

    kernel = Kernel()
    p_tr, p_out = program(prg_inp)
    k_tr, k_out = kernel(p_tr, p_out)
    # TODO: add some checks here

    forward = Forward(kernel, program)
    f_tr, f_out = forward(prg_inp)

    assert all(map(lambda ab: torch.allclose(ab[0], ab[1], atol=2.5), zip(f_out, k_out)))

def test_reverse():
    program = Simple()
    prg_inp = torch.ones([program.out_dim])

    kernel = Kernel()
    p_tr, p_out = program(prg_inp)
    k_tr, k_out = kernel(p_tr, p_out)
    # TODO: add some checks here

    reverse = Reverse(program, kernel)
    r_tr = reverse(prg_inp)

    # TODO: better tests
