import torch
from combinators.inference import *
from combinators.program import *
from combinators.densities import *
from combinators.densities.kernels import *

import combinators.debug as debug
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
    debug.seed(7)
    prg = Compose(q1=Simple1(), q2=Simple3())
    debug.seed(6)

    for s in range(1000):
        debug.seed(s)
        out = prg(_debug=True)

        assert set(out.trace.keys()) == {'x_1', 'x_2', 'x_3', 'z_1', 'z_2', 'z_3'}
        assert torch.equal(out.q1_out.log_weight, out.q1_out.trace.log_joint(nodes={'x_1', 'x_2'}))
        assert torch.equal(out.q2_out.log_weight, out.q2_out.trace.log_joint(nodes={'x_3'}))
        assert torch.equal(out.log_weight, out.q1_out.log_weight + out.q2_out.log_weight)




def test_extend_unconditioned():
    prg = Extend(p=Simple2(), f=Simple4())
    out = prg(_debug=True)
    assert set(out.trace.keys()) == {'x_2', 'x_3', 'z_2', 'z_3'}
    assert set(out.trace_star.keys()) == {'z_1'}
    assert torch.equal(out.log_weight, out.p_out.log_weight + out.f_out.trace.log_joint())


def test_extend_conditioned():
    P = Extend(p=Simple2(), f=Simple4())
    Q = Compose(q1=Simple1(), q2=Simple3())
    q_out = Q()
    p_out = Condition(program=P, cond_trace=q_out.trace)()
    tau_0  = {'z_1', 'z_2', 'z_3', 'x_2', 'x_3', 'x_1'}
    assert set(q_out.trace.keys()) == tau_0

    tau_1  = {       'z_2', 'z_3', 'x_2', 'x_3'}
    assert set(p_out.p_out.trace.keys()) == tau_1

    tau_2  = {'z_1'                            }
    assert set(p_out.f_out.trace.keys()) == tau_2

    lw_1   = p_out.p_out.trace.log_joint()
    assert lw_1 == p_out.p_out.log_weight

    lw_2   = p_out.f_out.trace.log_joint()
    assert lw_2 == p_out.f_out.log_weight

    lu_2   = lw_2
    lw_out = lw_1 + lu_2
    assert lw_out == p_out.log_weight


def test_extend_propose():
    debug.seed(7)
    Q = Compose(q1=Simple1(), q2=Simple3())
    # P = Extend(p=Simple2(), f=Simple4())
    prg = Propose(p=None, q=Q)
    count = 0
    seeds = []
    out = prg(_debug=True)
    # ===========================================================
    q_out = out.q_out
    # p_out = out.p_out
    # tau_1  = {'z_1', 'z_2', 'z_3', 'x_2', 'x_3', 'x_1'}
    # assert set(q_out.trace.keys()) == tau_1
    #
    # tau_2  = {       'z_2', 'z_3', 'x_2', 'x_3'}
    # assert set(p_out.trace.keys()) == tau_2
    #
    # tau_star = {'z_1'                            }
    # assert set(p_out.trace_star.keys()) == tau_star


    for s in range(1000):
        debug.seed(s)
        out = prg()
        assert torch.equal(
            q_out.trace.log_joint(nodes={'x_2', 'x_3', 'x_1'}),
            q_out.trace.log_joint(nodes={'x_1'}) + q_out.trace.log_joint(nodes={'x_2', 'x_3'}))

    breakpoint();

    lw_2   = p_out.trace.log_joint() + p_out.trace_star.log_joint()
    assert lw_2 == p_out.log_weight

    lu_1   = q_out.trace.log_joint(nodes={'x_2', 'x_3', 'x_1'})
    lw_out = lw_1 + lw_2 - lu_1
    #      x_1 x_2 x_3  * z_1 z_2 z_3 x_2 x_3
    #  -------------------------------------------------
    #      x_1 x_2 x_3
    assert lw_out == p_out.log_weight
























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
