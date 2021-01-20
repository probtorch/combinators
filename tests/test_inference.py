#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.distributions as dist
import logging
from torch import Tensor
from probtorch.util import expand_inputs
from collections import namedtuple
from typeguard import typechecked
from tqdm import trange
from pytest import mark, fixture
from typing import Optional, Callable

from .utils import assert_empirical_marginal_mean_std, g
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators.debug import propagate, excise_state, excise, empirical_marginal_mean_std
from combinators.objectives import nvo_avo
from combinators.metrics import effective_sample_size, log_Z_hat, Z_hat
from combinators.densities import Normal, MultivariateNormal
from combinators.densities.kernels import NormalLinearKernel
from combinators.nnets import LinearMap
from combinators.types import get_shape_kwargs
from combinators.tensor.utils import thash, show
from combinators.inference import PCache, State, Condition, RequiresGrad
from combinators.stochastic import RandomVariable, Provenance
from combinators import Program, Kernel, Trace, Forward, Reverse, Propose

logger = logging.getLogger(__name__)

@fixture(autouse=True)
def scaffolding():
    torch.manual_seed(1)
    targets  = [Normal(loc=i, scale=1, name=g(i)) for i in range(1,4) ]
    forwards = [NormalLinearKernel(ext_from=g(i), ext_to=g(i+1)) for i in range(1,3) ]
    reverses = [NormalLinearKernel(ext_from=g(i+1), ext_to=g(i)) for i in range(1,3) ]
    yield targets, forwards, reverses


@mark.skip("no longer excising state")
def test_excise_state_2step(scaffolding):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses

    q12 = Forward(f12, g1)
    p21 = Reverse(g2, r21)
    extend12 = Propose(target=p21, proposal=q12)
    state12, lv12 = extend12(sample_shape=(5,1))
    _state12q = excise_state(state12.proposal)
    _state12p = excise_state(state12.target)

    for k, prg, cpy in [(g(1), state12.target, _state12p), (g(2), state12.target, _state12p), (g(2), state12.proposal, _state12q)]:
        assert k == k and prg is prg and prg.trace[k].value.requires_grad # k==k for debugging the assert
        assert cpy.trace[k].value is not prg.trace[k].value
        assert torch.equal(cpy.trace[k].value, prg.trace[k].value)
        assert not cpy.trace[k].value.requires_grad

    assert not state12.proposal.trace[g(1)].value.requires_grad

    q23 = Forward(f23, g2)
    p32 = Reverse(g3, r32)
    extend23 = Propose(target=p32, proposal=q23)
    state23, lv23 = extend23()
    _state23q = excise_state(state23.proposal)
    _state23p = excise_state(state23.target)

    for k, prg, cpy in [(g(2), state23.target, _state23p), (g(3), state23.target, _state23p), (g(3), state23.proposal, _state23q)]:
        assert k == k and prg is prg and prg.trace[k].value.requires_grad # k==k for debugging the assert
        assert cpy.trace[k].value is not prg.trace[k].value
        assert torch.equal(cpy.trace[k].value, prg.trace[k].value)
        assert not cpy.trace[k].value.requires_grad

    assert not state23.proposal.trace[g(2)].value.requires_grad


def excise_params(m):
    # because we know these are all scalars via LinearMap, we .item()
    return [excise(param).item() for param in m.parameters()]

def test_no_targets_params(scaffolding):
    targets, forwards, reverses = scaffolding

    extract_params = lambda xs: [list(x.parameters()) for x in xs]
    target_params = extract_params(targets)
    assert all(map(lambda xs: len(xs) == 0, target_params))


def with_shape_test(scaffolding, sample_shape, sample_dims):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses

    tr, _, o = g1(sample_shape=sample_shape)

    assert tr[g1.name].value.shape[sample_dims] == sample_shape[sample_dims]

    # Step 1
    q12 = Forward(f12, g1)
    p21 = Reverse(g2, r21)
    extend12 = Propose(target=p21, proposal=q12)
    state12, lv12 = extend12(sample_shape=sample_shape, sample_dims=sample_dims)

    assert lv12.shape == torch.Size([sample_shape[sample_dims]])

    all_values = [rv.value for rv in [*state12.proposal.trace.values(), *state12.target.trace.values()]]
    for t in all_values:
        assert t.shape[sample_dims] == sample_shape[sample_dims]

    loss12 = nvo_avo(lv12)
    assert loss12.shape == torch.Size([])

def test_sample_shape_dim0(scaffolding):
    with_shape_test(scaffolding, sample_shape=(5,1), sample_dims=0)
@mark.skip("too hard for now, not sure this has a huge payoff at the moment")
def test_sample_shape_dim1(scaffolding):
    with_shape_test(scaffolding, sample_shape=(1,5), sample_dims=1)
@mark.skip("too hard for now, not sure this has a huge payoff at the moment")
def test_sample_shape_dim2(scaffolding):
    with_shape_test(scaffolding, sample_shape=(5,), sample_dims=0)
@mark.skip("too hard for now, not sure this has a huge payoff at the moment")
def test_sample_shape_dim_m1(scaffolding):
    with_shape_test(scaffolding, sample_shape=(5,), sample_dims=-1)

def test_disjoint_computation_graphs_if_backprop_on_step1(scaffolding):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses
    lr=1e-2

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=lr)
    f12_params0, f23_params0 = initial_f = excise_params(f12), excise_params(f23)
    r21_params0, r32_params0 = initial_r = excise_params(r21), excise_params(r32)


    # Step 1
    q12 = Forward(f12, g1)
    p21 = Reverse(g2, r21)
    extend12 = Propose(target=p21, proposal=q12)

    stat12, lv12 = extend12(sample_shape=(5,1), sample_dims=0)
    loss12 = nvo_avo(lv12)

    # Step 2
    q23 = Forward(f23, g2)
    p32 = Reverse(g3, r32)
    extend23 = Propose(target=p32, proposal=q23)

    state23, lv23 = extend23(sample_shape=(5,1), sample_dims=0)
    loss23 = nvo_avo(lv23, sample_dims=0).mean()

    # backward on step 1
    loss12.backward()

    for param in [*f12.parameters(), *r21.parameters()]:
        assert param.grad is not None and param.grad.item() != 0

    for param in [*f23.parameters(), *r32.parameters()]:
        assert param.grad is None

    optimizer.step()

    f12_params1, f23_params1 = post_f = [excise_params(f) for f in forwards]
    r21_params1, r32_params1 = post_r = [excise_params(r) for r in reverses]

    for p0, p1 in [*zip(f12_params0, f12_params1), *zip(r21_params0, r21_params1)] :
        assert p0 != p1

    for p0, p1 in [*zip(f23_params0, f23_params1), *zip(r32_params0, r32_params1)] :
        assert p0 == p1

def test_disjoint_computation_graphs_if_backprop_on_step2(scaffolding):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses
    lr=1e-2

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=lr)
    f12_params0, f23_params0 = excise_params(f12), excise_params(f23)
    r21_params0, r32_params0 = excise_params(r21), excise_params(r32)

    initial_forward_marginal2 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1), marginalize=True)
    initial_loc2, initial_cov2 = initial_forward_marginal2.loc.item(), initial_forward_marginal2.covariance_matrix.item()
    initial_forward_marginal3 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1), marginalize=True)
    initial_loc3, initial_cov3 = initial_forward_marginal3.loc.item(), initial_forward_marginal3.covariance_matrix.item()

    g1_prv_tr, _, _ = g1(sample_shape=(5,1))

    # Step 1
    q12 = Forward(f12, Condition(g1, g1_prv_tr, requires_grad=RequiresGrad.NO))
    p21 = Reverse(g2, r21)
    extend12 = Propose(target=p21, proposal=q12)
    state12, lv12 = extend12(sample_shape=(5,1), sample_dims=0)
    loss12 = nvo_avo(lv12)

    # Step 2
    q23 = Forward(f23, g2)
    p32 = Reverse(g3, r32)
    extend23 = Propose(target=p32, proposal=q23)
    state23, lv23 = extend23(sample_shape=(5,1), sample_dims=0)
    loss23 = nvo_avo(lv23, sample_dims=0).mean()

    # backward on step 2
    loss23.backward()

    for param in [*f12.parameters(), *r21.parameters()]:
        assert param.grad is None

    for param in [*f23.parameters(), *r32.parameters()]:
        assert param.grad is not None and param.grad.item() != 0

    optimizer.step()

    f12_params1, f23_params1 = post_f = [excise_params(f) for f in forwards]
    r21_params1, r32_params1 = post_r = [excise_params(r) for r in reverses]

    for p0, p1 in [*zip(f12_params0, f12_params1), *zip(r21_params0, r21_params1)] :
        assert p0 == p1

    for p0, p1 in [*zip(f23_params0, f23_params1), *zip(r32_params0, r32_params1)] :
        assert p0 != p1

def test_disjoint_computation_graphs_if_backprop_on_step2_in_run(scaffolding):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses
    lr=1e-2
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=lr)

    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []
    lvs_all = []

    num_steps = 100
    num_samples = 100

    initial_forward_marginal2 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1), marginalize=True)
    initial_loc2, initial_cov2 = initial_forward_marginal2.loc.item(), initial_forward_marginal2.covariance_matrix.item()

    initial_forward_marginal3 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1), marginalize=True)
    initial_loc3, initial_cov3 = initial_forward_marginal3.loc.item(), initial_forward_marginal3.covariance_matrix.item()

    with trange(num_steps) as bar:
        for i in bar:
            optimizer.zero_grad()
            q0 = targets[0]
            p_prv_tr, _, _ = q0(sample_shape=(num_samples, 1))

            lvs = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                # NVI-specific conditioning step
                q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO))
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state, lv = extend(sample_shape=(num_samples, 1), sample_dims=0)

                # setup for next step
                p_prv_tr = state.target.trace
                lvs.append(lv)
                if p.name == "g3":
                    # only run for step 1
                    # import ipdb; ipdb.set_trace();

                    loss = nvo_avo(lv, sample_dims=0).mean()

            lvs_ten = torch.stack(lvs, dim=0)
            lvs_all.append(lvs_ten)

            loss.backward()

            optimizer.step()
            # import ipdb; ipdb.set_trace();
            # report(lvs_all)


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
            # report(lvs_all)

    with torch.no_grad():
        analytic_forward_marginal2 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1), marginalize=True)
        analytic_loc2, analytic_cov2 = analytic_forward_marginal2.loc.item(), analytic_forward_marginal2.covariance_matrix.item()

        analytic_forward_marginal3 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1), marginalize=True)
        analytic_loc3, analytic_cov3 = analytic_forward_marginal3.loc.item(), analytic_forward_marginal3.covariance_matrix.item()

        assert analytic_loc2 == initial_loc2 and analytic_cov2 == initial_cov2
        assert g3.dist.loc.item() == 3. and abs(analytic_loc3 - g3.dist.loc.item()) != 0.


def test_training_run_full(scaffolding):
    targets, forwards, reverses = scaffolding
    g1, g2, g3 = targets
    f12, f23   = forwards
    r21, r32   = reverses
    lr=1e-2
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=lr)

    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []
    lvs_all = []

    num_steps = 1000
    num_samples = 100
    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, _, _ = q0(sample_shape=(num_samples, 1))
            loss = torch.zeros([1])

            lvs = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO))
                p_ext = Reverse(p, rev)
                extend_argument = Propose(target=p_ext, proposal=q_ext)
                state, lv = extend_argument(sample_shape=(num_samples, 1), sample_dims=0)
                lvs.append(lv)

                # setup for next step
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
        lvs = torch.stack(lvs_all, dim=0)
        lws = torch.cumsum(lvs, dim=1)
        ess = effective_sample_size(lws, sample_dims=0)
        print(ess)
        # FIXME: ESS looks funny. I think I am doing it wrong?
        analytic_forward_marginal2 = propagate(N=g1.as_dist(as_multivariate=True), F=f12.weight(), t=f12.bias(), B=torch.eye(1), marginalize=True)
        assert (g2.loc - analytic_forward_marginal2.loc).item() < 0.01
        analytic_forward_marginal3 = propagate(N=g2.as_dist(as_multivariate=True), F=f23.weight(), t=f23.bias(), B=torch.eye(1), marginalize=True)
        assert (g3.loc - analytic_forward_marginal3.loc).item() < 0.01
