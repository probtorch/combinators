import torch
from combinators.inference import *
from combinators.program import *
from combinators.densities import *
from combinators.densities.kernels import *

from combinators.stochastic import *
from combinators.utils import *

class Simple1(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_1 = trace.normal(loc=torch.ones(1), scale=torch.ones(1), name="z_1")
        z_2 = trace.normal(loc=torch.ones(2), scale=torch.ones(2), name="z_2")

        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_1, name="x_1")
        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_2, name="x_2")

class Simple2(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_2 = trace.normal(loc=torch.ones(2), scale=torch.ones(2), name="z_2")
        z_3 = trace.normal(loc=torch.ones(3), scale=torch.ones(3), name="z_3")

        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_2, name="x_2")
        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_3, name="x_3")

def test_run_a_primitive_program():
    s1 = Simple1()
    s1_out = s1()

def test_cond_eval():
    s1 = Simple1()
    s1_out = s1()

    s2 = Condition(program=Simple2(), cond_trace=s1_out.trace)
    s2_out = s2()

    assert len(set({'z_1', 'x_1'}).intersection(set(s2_out.trace.keys()))) == 0
