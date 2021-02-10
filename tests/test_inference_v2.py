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
        z_2 = trace.normal(loc=torch.ones(1)*2, scale=torch.ones(1)*2, name="z_2")

        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_1, name="x_1")
        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_2, name="x_2")

class Simple2(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_2 = trace.normal(loc=torch.ones(1)*2, scale=torch.ones(1)*2, name="z_2")
        z_3 = trace.normal(loc=torch.ones(1)*3, scale=torch.ones(1)*3, name="z_3")

        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_2, name="x_2")
        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_3, name="x_3")

class Simple3(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_3 = trace.normal(loc=torch.ones(1)*3, scale=torch.ones(1)*3, name="z_3")
        trace.normal(loc=torch.ones(1), scale=torch.ones(1), value=z_3, name="x_3")

class Simple4(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_1 = trace.normal(loc=torch.ones(1), scale=torch.ones(1), name="z_1")

def test_run_a_primitive_program():
    s1 = Simple1()
    s1_out = s1()

def test_cond_eval():
    s1 = Simple1()
    s1_out = s1()

    s2 = Condition(program=Simple2(), cond_trace=s1_out.trace)
    s2_out = s2()

    rho_f_addrs = {'x_2', 'x_3', 'z_2', 'z_3'}
    tau_f_addrs = {'z_2', 'z_3'}
    tau_p_addrs = {'z_1', 'z_2'}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    expected = s2_out.trace.log_joint(nodes=nodes)
    assert ( expected == s2_out.log_weight ).all()

    assert len(set({'z_1', 'x_1'}).intersection(set(s2_out.trace.keys()))) == 0

def test_propose():
    prg = Propose(p=Simple2(), q=Simple1())
    out = prg(_debug=True)

    rho_f_addrs = {'x_2', 'x_3', 'z_2', 'z_3'}
    tau_f_addrs = {'z_2', 'z_3'}
    tau_p_addrs = {'z_1', 'z_2'}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)

    expected = out.p_out.trace.log_joint(nodes=nodes)

    assert ( expected == out.p_out.log_weight ).all()
    assert torch.equal(out.q_out.log_weight + out.lv, out.log_weight)


def test_compose():
    prg = Compose(q1=Simple1(), q2=Simple3())
    out = prg(_debug=True)
    assert set(out.trace.keys()) == {'x_1', 'x_2', 'x_3', 'z_1', 'z_2', 'z_3'}
    assert torch.equal(out.log_weight, out.q1_out.log_weight + out.q2_out.log_weight)


def test_extend():
    prg = Extend(p=Simple2(), f=Simple4())
    out = prg(_debug=True)
    assert set(out.trace.keys()) == {'x_2', 'x_3', 'z_2', 'z_3'}
    assert set(out.trace_star.keys()) == {'z_1'}
    assert torch.equal(out.log_weight, out.p_out.log_weight + out.f_out.trace.log_joint())


def test_extend_propose():
    P = Extend(p=Simple2(), f=Simple4())
    Q = Compose(q1=Simple1(), q2=Simple3())
    prg = Propose(p=P, q=Q)

    out = prg()
    assert set(out.trace.keys()) == {'x_2', 'x_3', 'z_2', 'z_3'}

    rho_f_addrs = {'x_2', 'x_3', 'z_2', 'z_3'}
    tau_f_addrs = {'z_2', 'z_3'}
    tau_p_addrs = {'z_1', 'z_2'}
    u1_nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    assert torch.equal(out.q_out.trace['z_1'].value, out.p_out.f_out.trace['z_1'].value)
    # should hold for ever value

    naive_log_weight = trace_utils.copytraces(out.p_out.p_out.trace, out.p_out.f_out.trace).log_joint()
    naive_log_weight =- out.q_out.trace.log_joint() - out.q_out.trace['x_1'].log_prob
    breakpoint();

    assert torch.equal(out.log_weight, out.p_out.log_weight + out.q_out.log_weight - (out.q_out.trace.log_joint(nodes=u1_nodes) - out.trace_star.log_joint()))
    breakpoint();
    print()
