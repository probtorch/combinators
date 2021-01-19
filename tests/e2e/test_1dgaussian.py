import torch
import torch.nn as nn
import logging
from torch import Tensor
from torch import distributions
from combinators.densities import Normal
from combinators.densities.kernels import NormalLinearKernel
from collections import namedtuple
from typeguard import typechecked
from tqdm import trange
from pytest import mark, fixture
from typing import Optional

from ..utils import assert_empirical_marginal_mean_std, Tolerance, Params

import combinators.trace.utils as trace_utils
from combinators.metrics import effective_sample_size
from combinators.debug import propagate
from combinators.objectives import nvo_avo
from combinators.tensor.utils import thash, show, autodevice, kw_autodevice
from combinators.inference import PCache # temporary
from combinators.stochastic import RandomVariable, Provenance
from combinators import Program, Kernel, Trace, Forward, Reverse, Propose

logger = logging.getLogger(__name__)

eval_mean_std = assert_empirical_marginal_mean_std

@typechecked
class MLPKernel(Kernel):
    def __init__(self, dim_hidden, ext_name):
        super().__init__()
        self.ext_name = ext_name
        self.net = nn.Sequential(
            nn.Linear(1, dim_hidden), nn.Sigmoid(),
            nn.Linear(dim_hidden, dim_hidden), nn.Sigmoid(),
            nn.Linear(dim_hidden, 1))

    def apply_kernel(self, trace, cond_trace, obs):
        return trace.normal(loc=self.net(obs.detach()),
                            scale=torch.ones(1),
                            value=None if self.ext_name not in cond_trace else cond_trace[self.ext_name].value,
                            name=self.ext_name)


@fixture(autouse=True)
def seed():
    torch.manual_seed(1)


def test_forward(seed):
    g = Normal(loc=0, scale=1, name="g")
    fwd = MLPKernel(dim_hidden=4, ext_name="fwd")

    ext = Forward(fwd, g)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

def test_forward_forward(seed):
    g0 = Normal(loc=0, scale=1, name="g0")
    f01 = MLPKernel(dim_hidden=4, ext_name="g1")
    f12 = MLPKernel(dim_hidden=4, ext_name="g2")

    ext = Forward(f12, Forward(f01, g0))
    ext()

def test_reverse(seed):
    g = Normal(loc=0, scale=1, name="g")
    rev = MLPKernel(dim_hidden=4, ext_name="rev")

    ext = Reverse(g, rev)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

def test_propose_values(seed):
    q = Normal(loc=4, scale=1, name="z_0")
    p = Normal(loc=0, scale=4, name="z_1")
    fwd = MLPKernel(dim_hidden=4, ext_name="z_1")
    rev = MLPKernel(dim_hidden=4, ext_name="z_0")
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5)
    assert len(list(p.parameters())) == 0
    assert len(list(q.parameters())) == 0
    fwd_hashes_0 = [thash(t) for t in fwd.parameters()]
    rev_hashes_0 = [thash(t) for t in rev.parameters()]

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    extend = Propose(target=p_ext, proposal=q_ext)

    _, log_weights = extend()

    assert isinstance(log_weights, Tensor)

    cache = extend._cache
    # import ipdb; ipdb.set_trace();

    for k in ['z_0', 'z_1']:
        assert torch.equal(cache.proposal.trace[k].value, cache.target.trace[k].value)

    loss = nvo_avo(log_weights, sample_dims=0).mean()
    loss.backward()

    optimizer.step()
    fwd_hashes_1 = [thash(t) for t in fwd.parameters()]
    rev_hashes_1 = [thash(t) for t in rev.parameters()]

    assert any([l != r for l, r in zip(fwd_hashes_0, fwd_hashes_1)])
    assert any([l != r for l, r in zip(rev_hashes_0, rev_hashes_1)])

def test_propose_gradients(seed):
    q = Normal(loc=4, scale=1, name="z_0")
    p = Normal(loc=0, scale=4, name="z_1")
    fwd = MLPKernel(dim_hidden=4, ext_name="z_1")
    rev = MLPKernel(dim_hidden=4, ext_name="z_0")
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5)

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    extend = Propose(target=p_ext, proposal=q_ext)

    _, log_weights = extend()
    cache = extend._cache

    for k, prg in [("z_1", cache.target), ("z_0", cache.target), ("z_1", cache.proposal)]:
        assert k == k and prg is prg and prg.trace[k].value.requires_grad # k==k for debugging the assert

    assert not cache.proposal.trace["z_0"].value.requires_grad

def test_1step_avo(seed):
    """ The VAE test. At one step no need for any detaches. """

    target_params, proposal_params = all_params = [Params(4, 1), Params(1, 4)]
    target,        proposal        = [Normal(*p, name=f'z_{p.loc}') for p in all_params]
    fwd, rev = [MLPKernel(dim_hidden=4, ext_name=f'z_{ext_mean}') for ext_mean in [4, 1]]

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [proposal, target, fwd, rev]], lr=0.01)

    num_steps = 100
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            optimizer.zero_grad()
            q_ext = Forward(fwd, proposal)
            p_ext = Reverse(target, rev)
            extend = Propose(target=p_ext, proposal=q_ext)

            _, log_weights = extend(sample_shape=(200,1), sample_dims=0)

            # proposal.clear_observations() # FIXME: this can be automated, but it would be nice to have more infrastructure around observations
            loss = nvo_avo(log_weights).mean()

            loss.backward()

            optimizer.step()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if num_steps <= 100:
               loss_avgs.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_avgs.append(loss_avg)
    with torch.no_grad():
        assert_empirical_marginal_mean_std(lambda: Forward(fwd, proposal)()[-1], target_params, Tolerance(loc=0.15, scale=0.15))


def test_2step_avo(seed):
    """
    2-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).

    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
    """
    g1, g2, g3 = targets = [Normal(loc=i, scale=1, name=f"z_{i}") for i in range(1,4)]
    f12, f23 = forwards = [NormalLinearKernel(ext_from=f"z_{i}", ext_to=f"z_{i+1}").to(autodevice()) for i in range(1,3)]
    r21, r32 = reverses = [NormalLinearKernel(ext_from=f"z_{i+1}", ext_to=f"z_{i}").to(autodevice()) for i in range(1,3)]

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = 2000
    loss_ct, loss_sum, loss_all = 0, 0.0, []
    lvs_all = []
    sample_shape=(100,1)

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros([1], **kw_autodevice())

            lvs = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend_argument = Propose(target=p_ext, proposal=q_ext)
                # state, lv = extend_argument(sample_shape=sample_shape) # TODO
                state, lv = extend_argument(sample_shape=sample_shape, sample_dims=0)
                q.clear_observations()

                lvs.append(lv)

                p_prv_tr = state.target.trace
                loss += nvo_avo(lv, sample_dims=0).mean()

            lvs_ten = torch.stack(lvs, dim=0)
            lvs_all.append(lvs_ten)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0

    with torch.no_grad():
        # report.sparkline(loss_avgs)
        lvs = torch.stack(lvs_all, dim=0)
        lws = torch.cumsum(lvs, dim=1)
        ess = effective_sample_size(lws)
        # import matplotlib.pyplot as plt
        # plt.plot(ess)
        # plt.savefig("fig.png")

        # This is the analytic marginal for the forward kernel
        out12 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1, **kw_autodevice()), marginalize=True)
        print(out12.loc);
        out23 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1, **kw_autodevice()), marginalize=True)
        print(out23.loc);

        tol = Tolerance(loc=0.15, scale=0.15)
        assert abs((out12.loc - g2.dist.loc).item()) < tol.loc
        assert abs((out23.loc - g3.dist.loc).item()) < tol.loc

        tr, out = g1(sample_shape=(200,1))
        assert abs(out.mean().item() - 1) < tol.loc
        tr, out = f12(tr, out)
        assert abs(out.mean().item() - 2) < tol.loc

        pre2 = Forward(f12, g1)
        tr, out = pre2(sample_shape=(200, 1))
        assert abs(out.mean().item() - 2) < tol.loc

        tr, out = g2(sample_shape=(200,1))
        assert abs(out.mean().item() - 2) < tol.loc
        tr, out = f23(tr, out)
        assert abs(out.mean().item() - 3) < tol.loc

        pre3 = Forward(f23, g2)
        tr, out = pre3(sample_shape=(200,1))
        assert abs(out.mean().item() - 3) < tol.loc

        predict_g1_to_g2 = lambda: pre2()[1]
        predict_g2_to_g3 = lambda: pre3()[1]
        predict_g1_to_g3 = lambda: Forward(f23, Forward(f12, g1))()[1]

        assert_empirical_marginal_mean_std(predict_g1_to_g2, Params(loc=2, scale=1), tol)
        assert_empirical_marginal_mean_std(predict_g2_to_g3, Params(loc=3, scale=1), tol)
        assert_empirical_marginal_mean_std(predict_g1_to_g3, Params(loc=3, scale=1), tol)


def test_4step_avo(seed):
    """
    4-step NVI-sequential: 8 intermediate densities
    """
    g1, g2, g3, g4, g5 = targets = [Normal(loc=i, scale=1, name=f"z_{i}") for i in range(1,6)]
    f12, f23, f34, f45 = forwards = [NormalLinearKernel(ext_from=f"z_{i}", ext_to=f"z_{i+1}") for i in range(1,5)]
    r21, r32, r43, r54 = reverses = [NormalLinearKernel(ext_from=f"z_{i+1}", ext_to=f"z_{i}") for i in range(1,5)]
    assert r21.ext_to == "z_1"
    assert f12.ext_to == "z_2"
    assert r54.ext_to == "z_4"
    assert f45.ext_to == "z_5"

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = 4000
    sample_shape=(100,1)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)
            loss = torch.zeros(1)

            lvs = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend_argument = Propose(target=p_ext, proposal=q_ext)
                # state, lv = extend_argument(sample_shape=sample_shape) # TODO
                state, lv = extend_argument(sample_shape=sample_shape, sample_dims=0)
                q.clear_observations()
                p.clear_observations()

                lvs.append(lv)

                p_prv_tr = state.target.trace
                loss += nvo_avo(lv, sample_dims=0).mean()


            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0

    with torch.no_grad():
        tol = Tolerance(loc=0.15, scale=0.15)

        out12 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1), marginalize=True)
        out23 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1), marginalize=True)
        out34 = propagate(N=g3.as_dist(as_multivariate=True), F=f34.weight(), t=f34.bias(), B=torch.eye(1), marginalize=True)
        out45 = propagate(N=g4.as_dist(as_multivariate=True), F=f45.weight(), t=f45.bias(), B=torch.eye(1), marginalize=True)
        for (analytic, target_loc) in zip([out12, out23, out34, out45], range(2,6)):
            assert (target_loc - analytic.loc.item()) < tol.loc

        predict_g2_chain = lambda: Forward(f12, g1)()[1]
        predict_g3_chain = lambda: Forward(f23, Forward(f12, g1))()[1]
        predict_g4_chain = lambda: Forward(f34, Forward(f23, Forward(f12, g1)))()[1]
        predict_g5_chain = lambda: Forward(f45, Forward(f34, Forward(f23, Forward(f12, g1))))()[1]

        eval_mean_std(predict_g2_chain, Params(loc=2, scale=1), tol)
        eval_mean_std(predict_g3_chain, Params(loc=3, scale=1), tol)
        eval_mean_std(predict_g4_chain, Params(loc=4, scale=1), tol)
        eval_mean_std(predict_g5_chain, Params(loc=5, scale=1), tol)

        predict_g2 = lambda: Forward(f12, g1)()[1]
        predict_g3 = lambda: Forward(f23, g2)()[1]
        predict_g4 = lambda: Forward(f34, g3)()[1]
        predict_g5 = lambda: Forward(f45, g4)()[1]

        eval_mean_std(predict_g2, Params(loc=2, scale=1), tol)
        eval_mean_std(predict_g3, Params(loc=3, scale=1), tol)
        eval_mean_std(predict_g4, Params(loc=4, scale=1), tol)
        eval_mean_std(predict_g5, Params(loc=5, scale=1), tol)
