#!/usr/bin/env python
#
import torch
import torch.nn as nn
import logging
from torch import Tensor
from torch import distributions
from torch.utils.tensorboard import SummaryWriter
from pytest import fixture, mark
from tqdm import trange

import combinators.trace.utils as trace_utils
from combinators.tensor.utils import kw_autodevice
from combinators import Forward, Reverse, Propose, Kernel
from combinators.metrics import effective_sample_size
from combinators.debug import propagate
from combinators.nnets import LinearMap, ResMLPJ
from combinators.objectives import nvo_rkl
from combinators.densities import MultivariateNormal, Tempered, RingGMM
from combinators.densities.kernels import MultivariateNormalLinearKernel, MultivariateNormalKernel, IdentityKernel


@fixture(autouse=True)
def seed():
    torch.manual_seed(1)

def test_tempered_redundant(seed):
    # hyperparams
    sample_shape = (13,)

    # Setup
    g0 = MultivariateNormal(name='g0', loc=torch.ones(2), cov=torch.eye(2)**2)
    gK = MultivariateNormal(name='g2', loc=torch.ones(2), cov=torch.eye(2)**2)
    halfway = Tempered('g1', g0, gK, torch.tensor([0.5]))
    f01, f12 = [MultivariateNormalLinearKernel(ext_from=f'g{i}', ext_to=f'g{i+1}', loc=torch.ones(2), cov=torch.eye(2)) for i in range(0,2)]
    r10, r21 = [MultivariateNormalLinearKernel(ext_from=f'g{i+1}', ext_to=f'g{i}', loc=torch.ones(2), cov=torch.eye(2)) for i in range(0,2)]

    # NVI labels:
    p_prv_tr, _ = g0(sample_shape=sample_shape)
    g0.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
    q1_ext = Forward(f01, g0)
    p1_ext = Reverse(halfway, r10)
    extend1 = Propose(target=p1_ext, proposal=q1_ext)
    state1, lv1 = extend1(sample_shape=sample_shape, sample_dims=0)

    p_prv_tr = state1.target.trace
    g0.clear_observations()

    halfway.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
    q2_ext = Forward(f12, halfway)
    p2_ext = Reverse(gK, r21)
    extend2 = Propose(target=p2_ext, proposal=q2_ext)
    state2, lv2 = extend2(sample_shape=sample_shape, sample_dims=0)

def test_tempered_redundant_trivial(seed):
    # hyperparams
    sample_shape = (100,)
    num_iterations = 1000

    # Setup
    g0 = MultivariateNormal(name='g0', loc=torch.ones(2), cov=torch.eye(2)**2)
    gK = MultivariateNormal(name='g2', loc=torch.ones(2)*3, cov=torch.eye(2)**2)
    halfway = Tempered('g1', g0, gK, torch.tensor([0.5]))
    forwards = [MultivariateNormalLinearKernel(ext_from=f'g{i}', ext_to=f'g{i+1}', loc=torch.ones(2), cov=torch.eye(2)) for i in range(0,2)]
    reverses = [MultivariateNormalLinearKernel(ext_from=f'g{i+1}', ext_to=f'g{i}', loc=torch.ones(2), cov=torch.eye(2)) for i in range(0,2)]
    targets = [g0, halfway, gK]
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-4)

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_iterations) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros(1, **kw_autodevice())
            lw, lvs = torch.zeros(sample_shape, **kw_autodevice()), []
            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q, _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)

                p_prv_tr = state.target.trace
                p.clear_observations()
                q.clear_observations()

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
                lvs.append(lv)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            writer.add_scalar('Loss/train', loss_scalar, i)
            loss_sum += loss_scalar
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               loss_ct, loss_sum  = 0, 0.0
               bar.set_postfix_str(loss_template)


    with torch.no_grad():
        f12, f23 = forwards
        analytic2 = propagate(N=g0.dist, F=f12.weight(), t=f12.bias(), B=g0.dist.covariance_matrix, marginalize=True)
        # TODO: add some tests?


def test_annealing_path_tempered_normals(seed):
    num_targets = 3
    g0 = MultivariateNormal(name='g0', loc=torch.zeros(2), cov=torch.eye(2)**2)
    gK = MultivariateNormal(name=f'g{num_targets}', loc=torch.ones(2)*8, cov=torch.eye(2)**2)
    forwards = [MultivariateNormalLinearKernel(ext_name=f'g{i}', loc=torch.ones(2)*i, cov=torch.eye(2)) for i in range(1,num_targets+1)]
    reverses = [MultivariateNormalLinearKernel(ext_name=f'g{i}', loc=torch.ones(2)*i, cov=torch.eye(2)) for i in range(0,num_targets)]

    betas = torch.arange(0., 1., 1./num_targets)[1:] # g_0 is beta=0

    path = [Tempered(f'g{k}', g0, gK, beta) for k, beta in zip(range(1,num_targets), betas)]
    path = [g0] + path + [gK]
    targets = path

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = 10
    sample_shape = (7,)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)
            loss = torch.zeros(1)

            lvs = []
            lw = torch.zeros(sample_shape)

            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)

                # FIXME: because p_prv_tr is not eliminating the previous trace, the trace is cumulativee but removing grads leaves backprop unaffected
                p_prv_tr = state.target.trace
                q.clear_observations()
                p.clear_observations()
                assert set(p_prv_tr.keys()) == { f'g{k+1}' }

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
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

def mk_kernel(target:int, std:float, num_hidden:int):
    embedding_dim = 2
    return MultivariateNormalKernel(
        ext_name=f'g_{target}',
        loc=torch.zeros(2),
        cov=torch.eye(2)*std**2,
        net=ResMLPJ(
            dim_in=2,
            dim_hidden=num_hidden,
            dim_out=embedding_dim,
        ))

def mk_model(num_targets:int):
    proposal_std = 1.0
    g_0 = MultivariateNormal(name='g_0', loc=torch.zeros(2), cov=torch.eye(2)*proposal_std**2)
    g_K = RingGMM(scale=8, count=8, name=f"g_{num_targets - 1}")

    # Make an annealing path
    betas = torch.arange(0., 1., 1./(num_targets - 1))[1:] # g_0 is beta=0
    path = [Tempered(f'g_{k}', g_0, g_K, beta) for k, beta in zip(range(1,num_targets-1), betas)]
    path = [g_0] + path + [g_K]
    assert len(path) == num_targets # sanity check that the betas line up

    num_kernels = num_targets - 1
    target_ixs = [ix for ix in range(0, num_targets)]
    mk_kernels = lambda shift_target: [
        mk_kernel(target=shift_target(ix),
                  std=shift_target(ix)+1.0,
                  num_hidden=64
                  ) for ix in target_ixs[:num_kernels]
    ]

    return dict(
        targets=path,
        forwards=mk_kernels(lambda ix: ix+1),
        reverses=mk_kernels(lambda ix: ix),
    )

def test_annealing_path(seed):
    K = 8
    out = mk_model(K)
    targets, forwards, reverses = [out[n] for n in ['targets', 'forwards', 'reverses']]
    assert all([len(list(t.parameters())) == 0 for t in targets])
    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)
    num_samples = 7
    num_steps = 13
    sample_shape=(num_samples,)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)
            loss = torch.zeros(1)

            lvs = []
            lw = torch.zeros(sample_shape)

            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q, _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)

                # FIXME: because p_prv_tr is not eliminating the previous trace, the trace is cumulativee but removing grads leaves backprop unaffected
                p_prv_tr = state.target.trace
                q.clear_observations()
                p.clear_observations()
                assert set(p_prv_tr.keys()) == { f'g_{k+1}' }

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g_{k}'], state.target.trace[f'g_{k+1}'])
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
