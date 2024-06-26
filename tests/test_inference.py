# Copyright 2021-2024 Northeastern University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from combinators.inference import Program, Compose, Extend, Propose
import combinators.utils as debug
from pytest import mark


class Simple1(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_1 = trace.normal(
            loc=torch.ones(1), scale=torch.ones(1), reparameterized=False, name="z_1"
        )
        z_2 = trace.normal(
            loc=torch.ones(1) * 2,
            scale=torch.ones(1) * 2,
            reparameterized=False,
            name="z_2",
        )

        trace.normal(
            loc=torch.ones(1),
            scale=torch.ones(1),
            value=z_1,
            reparameterized=False,
            name="x_1",
        )
        trace.normal(
            loc=torch.ones(1),
            scale=torch.ones(1),
            value=z_2,
            reparameterized=False,
            name="x_2",
        )
        return torch.tensor(3.0)


class Simple2(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c=None):
        z_2 = trace.normal(
            loc=torch.ones(1) * 2,
            scale=torch.ones(1) * 2,
            reparameterized=False,
            name="z_2",
        )
        z_3 = trace.normal(
            loc=torch.ones(1) * 3,
            scale=torch.ones(1) * 3,
            reparameterized=False,
            name="z_3",
        )

        trace.normal(
            loc=torch.ones(1),
            scale=torch.ones(1),
            value=z_2,
            reparameterized=False,
            name="x_2",
        )
        trace.normal(
            loc=torch.ones(1),
            scale=torch.ones(1),
            value=z_3,
            reparameterized=False,
            name="x_3",
        )
        return torch.tensor(1.0)


class Simple3(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c):
        z_3 = trace.normal(
            loc=torch.ones(1) * c,
            scale=torch.ones(1) * 3,
            reparameterized=False,
            name="z_3",
        )
        trace.normal(
            loc=torch.ones(1),
            scale=torch.ones(1),
            value=z_3,
            reparameterized=False,
            name="x_3",
        )


class Simple4(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c):
        z_1 = trace.normal(
            loc=torch.ones(1) * c,
            scale=torch.ones(1),
            reparameterized=False,
            name="z_1",
        )


# ===== #
# TESTS #
# ===== #


def test_run_a_primitive_program():
    s1_out = Simple1()(None)
    assert set(s1_out.trace.keys()) == {"z_1", "z_2", "x_1", "x_2"}
    assert s1_out.log_weight == s1_out.trace.log_joint(nodes={"x_1", "x_2"})


@mark.skip()
def test_cond_eval():
    s1_out = Simple1()(None)
    s2_out = Condition(program=Simple2(), cond_trace=s1_out.trace)(None)

    rho_f_addrs = {"x_2", "x_3", "z_2", "z_3"}
    tau_f_addrs = {"z_2", "z_3"}
    tau_p_addrs = {"z_1", "z_2"}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    lw_out = s2_out.trace.log_joint(nodes=nodes)

    assert (lw_out == s2_out.log_weight).all(), "lw_out"
    assert (
        len(set({"z_1", "x_1"}).intersection(set(s2_out.trace.keys()))) == 0
    ), "tr_out"


@mark.skip()
def test_propose():
    out = Propose(p=Simple2(), q=Simple1())(None, _debug=True)

    rho_q_addrs = {"x_1", "x_2", "z_1", "z_2"}
    tau_q_addrs = {"z_1", "z_2"}
    tau_p_addrs = {"z_2", "z_3"}
    nodes = rho_q_addrs - (tau_q_addrs - tau_p_addrs)

    assert set(out.q_out.trace.keys()) == {"z_1", "z_2", "x_1", "x_2"}
    assert set(out.p_out.trace.keys()) == {"z_2", "z_3", "x_2", "x_3"}

    # Compute stuff the same way as in the propose combinator to ensure numeric reproduceability
    lu_1 = out.q_out.trace.log_joint(nodes=nodes)
    lu_star = torch.zeros(1)
    lw_1 = out.q_out.log_weight
    lv = out.p_out.log_weight - (lu_1 + lu_star)
    lw_out = lw_1 + lv
    assert torch.equal(lw_out.squeeze(), out.log_weight), "lw_out"


def test_compose():
    out = Compose(q1=Simple1(), q2=Simple3())(None, _debug=True)

    assert set(out.trace.keys()) == {"x_1", "x_2", "x_3", "z_1", "z_2", "z_3"}
    assert torch.equal(
        out.q1_out.log_weight, out.q1_out.trace.log_joint(nodes={"x_1", "x_2"})
    )
    assert torch.equal(out.q2_out.log_weight, out.q2_out.trace.log_joint(nodes={"x_3"}))
    assert torch.equal(out.log_weight, out.q1_out.log_weight + out.q2_out.log_weight)


@mark.skip()
def test_extend_unconditioned():
    out = Extend(p=Simple2(), f=Simple4())(None, _debug=True)
    assert set(out.trace.keys()) == {"x_2", "x_3", "z_2", "z_3"}
    assert set(out.trace_star.keys()) == {"z_1"}
    assert torch.equal(
        out.log_weight, out.p_out.log_weight + out.f_out.trace.log_joint()
    )


@mark.skip()
def test_extend_conditioned():
    q_out = Compose(q1=Simple1(), q2=Simple3())(None, _debug=True)
    p_out = Condition(Extend(p=Simple2(), f=Simple4()), cond_trace=q_out.trace)(
        None, _debug=True
    )
    tau_0 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
    assert set(q_out.trace.keys()) == tau_0

    # Check Compose
    tau_1 = {"z_2", "z_3", "x_2", "x_3"}
    assert set(p_out.p_out.trace.keys()) == tau_1

    lw_1 = p_out.p_out.trace.log_joint(nodes=["z_2", "z_3", "x_2", "x_3"])
    lw_1_ = p_out.p_out.trace.log_joint()
    assert lw_1 == lw_1_
    assert lw_1 == p_out.p_out.log_weight, "lw_1"

    # Check Extend
    tau_2 = {"z_1"}
    assert set(p_out.f_out.trace.keys()) == tau_2

    lw_2 = p_out.f_out.trace.log_joint()
    assert lw_2 == p_out.f_out.log_weight, "lw_2"

    # Check actual weight computed by cond. extend
    lu_2 = lw_2
    lw_out = lw_1 + lu_2
    assert lw_out == p_out.log_weight, "lw_out"


@mark.skip()
def test_extend_propose():
    debug.seed(7)
    Q = Compose(q1=Simple1(), q2=Simple3())
    P = Extend(p=Simple2(), f=Simple4())
    out = Propose(p=P, q=Q)(None, _debug=True)

    # Test compose inside propose
    tau_1 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
    assert set(out.q_out.trace.keys()) == tau_1

    lw_1 = out.q_out.trace.log_joint(nodes={"x_1", "x_2"}) + out.q_out.trace.log_joint(
        nodes={"x_3"}
    )
    # FIXME: computing the log joint in on go returns different result
    # lw_1_ = out.q_out.trace.log_joint(nodes={'x_2', 'x_3', 'x_1'})
    # assert lw_1 == lw_1_
    assert lw_1 == out.q_out.log_weight

    # Test extend inside propose
    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    assert set(out.p_out.trace.keys()) == tau_2
    tau_star = {"z_1"}
    assert set(out.p_out.trace_star.keys()) == tau_star

    lw_2 = out.p_out.trace.log_joint(
        nodes={"z_2", "z_3", "x_2", "x_3"}
    ) + out.p_out.trace_star.log_joint(nodes={"z_1"})
    assert lw_2 == out.p_out.log_weight

    # Compute weight the same way as inside the propose combinator for reproduceability
    lu_1 = out.q_out.trace.log_joint(nodes={"x_2", "x_3", "x_1", "z_2", "z_3"})
    lu_star = out.p_out.trace_star.log_joint(nodes={"z_1"})
    lv = lw_2 - (lu_1 + lu_star)
    lw_out = lw_1 + lv
    #   (x_1 x_2 x_3) * ((z_2 z_3  *  x_2 x_3) * (z_1))
    #  ------------------------------------------------
    #  ((x_1 x_2 x_3  *   z_2 z_3)             * (z_1))
    assert lw_out == out.log_weight, "final is weight"
    print(lw_out)
