#!/usr/bin/env python3
from combinators.inference import Condition, Resample
import torch
import math
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt
from functools import partial

from combinators import adam, ppr
from combinators import autodevice, kw_autodevice
from combinators import nvo_rkl, nvo_avo
from combinators import Propose, Extend, Compose, Resample
from combinators import effective_sample_size, log_Z_hat
from combinators import Systematic
from combinators.inference import copytraces
from combinators.trace.utils import valeq

import experiments.visualize as V
from experiments.annealing.models import sample_along, paper_model

# TODO: delete and replace with traverse_proposals, below
def _get_stats(out, lw=[], loss=[], proposal_traces=[], target_traces=[], resample=False)->[Tensor]:
    if resample:
        out=out.q_out
    assert out.type == "Propose"

    _lw = [out.log_weight.detach().cpu()] + lw
    _loss = [out.loss.detach().cpu()] + loss
    _proposal_traces = [out.proposal_trace] + proposal_traces
    _target_traces = [out.target_trace] + target_traces

    if out.q_out.q1_out.type == "Propose" or (resample and out.q_out.q1_out.type == "Resample"):
        out = out.q_out.q1_out
        return _get_stats(out, lw=_lw, loss=_loss, resample=resample)
    elif out.q_out.q2_out.type == "Propose" or (resample and out.q_out.q2_out.type == "Resample"):
        out = out.q_out.q2_out
        return _get_stats(out, lw=_lw, loss=_loss, resample=resample)
    else:
        return (_lw, _loss, _proposal_traces, target_traces)
    #FIXME: It is this gd fn

def traverse_proposals(fn, out, memo=[])->[Tensor]:
    if out.pytype == Propose:
        return traverse_proposals(fn, out.q_out, memo + [fn(out)])
    elif out.pytype == Extend:
        raise ValueError("impossible! traverse proposal will never arrive here")
    elif out.pytype == Compose:
        valid_infs = [Compose, Propose, Resample]
        q1_is_inf = out.q1_out.pytype in valid_infs
        q2_is_inf = out.q2_out.pytype in valid_infs
        if not (q1_is_inf ^ q2_is_inf):
            return memo
        else:
            return traverse_proposals(fn, out.q1_out if q1_is_inf else out.q2_out, memo)
    elif out.pytype == Resample:
        return traverse_proposals(fn, out.q_out, memo)
    elif out.pytype == Condition:
        raise ValueError("impossible! traverse proposal will never arrive here")
    else:
        return memo

def get_stats(out):
    fn = lambda out: (out.log_weight, out.loss, out.proposal_trace, out.target_trace, out)
    ret = traverse_proposals(fn, out)
    lws, losses, proposal_trace, target_trace, outs = zip(*ret)
    return dict(lw=lws, loss=losses, proposal_trace=proposal_trace, target_trace=target_trace, out=outs)

def print_and_sum_loss(loss_fn, out, loss):
    step_loss = loss_fn(out)
    loss = loss + step_loss
    # step_loss = step_loss if not loss == 0. else step_loss.detach()
    # print(loss)
    return loss

def nvi_declarative(targets, forwards, reverses, loss_fn, sample_shape, batch_dim, sample_dims, resample=False):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(p=Extend(p, rev),
                    q=Compose(q, fwd),
                    loss_fn=partial(print_and_sum_loss, loss_fn), _debug=True, ix=k, loss0=torch.zeros(1, **kw_autodevice()))
        if resample:
            q = Resample(q)
    out = q(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
    return out

def compute_nvi_weight(proposal, target, forward, reverse, lw_in=0., sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False):
    g0_out = proposal(None, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    g0_rv = g0_out.trace[proposal.name]
    g0_lp = g0_rv.log_prob

    f12_out = forward({f'{proposal.name}': g0_rv.value}, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    f12_rv = f12_out.trace[target.name]
    f12_lp = f12_rv.log_prob

    target._cond_trace = f12_out.trace
    g1_out = target(None, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    g1_rv = g1_out.trace[target.name]
    g1_lp = g1_rv.log_prob

    reverse._cond_trace = g0_out.trace
    r21_out = reverse({f'{target.name}': g1_rv.value}, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    r21_rv = r21_out.trace[proposal.name]
    r21_lp = r21_rv.log_prob

    lw_out = (g1_lp + r21_lp) - (g0_lp + f12_lp) + lw_in
    lw_out_ = lw_out
    trace_out = g1_out.trace
    if resample:
        trace_out, lw_out = Systematic()(trace_out, lw_out, sample_dims=sample_dims, batch_dim=batch_dim)
    return lw_out, lw_out_, trace_out, g1_out, g0_out, f12_out, r21_out

def test_1_step_nvi(sample_shape=(10, 5), batch_dim=1, sample_dims=0):
    out = paper_model(2)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    seeds = torch.arange(100)
    for seed in seeds:
        torch.manual_seed(seed)
        out = nvi_declarative(targets, forwards, reverses, nvo_avo,
                              sample_shape, batch_dim=batch_dim, sample_dims=sample_dims)

        torch.manual_seed(seed)

        lw, _ = compute_nvi_weight(targets[0], targets[1], forwards[0], reverses[0], lw_in=0.,
                                   sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
        assert (lw == out.log_weight).all()

    # main(nvi_eager_resample, K=8, seed=1, eval_break=50, num_iterations=10000)

def test_K_step_nvi(K, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False, num_seeds=10):
    out = paper_model(K)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    seeds = torch.arange(num_seeds)
    for seed in seeds:
        torch.manual_seed(seed)
        out = nvi_declarative(targets, forwards, reverses, nvo_avo,
                              sample_shape, batch_dim=batch_dim, sample_dims=sample_dims,
                              resample=resample)
        stats_nvi_dec = get_stats(out)

        torch.manual_seed(seed)
        lw = 0.
        cond_trace = None
        cond_traces = [cond_trace]
        out_refs = [out]
        target_outs = []
        proposal_outs = []
        fwd_outs = []
        rev_outs = []
        lws_ = []
        lws = []
        for k in range(K):
            targets[k]._cond_trace = copytraces(cond_trace) if cond_trace is not None else None
            nvi_out  = compute_nvi_weight(targets[k], targets[k+1], forwards[k], reverses[k], lw_in=lw,
                                          sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim,
                                          resample=resample)
            lw, lw_, cond_trace, target_out, proposal_out, fwd_out, rev_out = nvi_out
            proposal_trace_manual = copytraces(fwd_out.trace, proposal_out.trace)
            target_trace_manual = copytraces(rev_out.trace, target_out.trace)

            lws.append(lw)
            lws_.append(lw_)
            cond_traces.append(cond_trace)
            target_outs.append(target_out)
            proposal_outs.append(proposal_out)
            fwd_outs.append(fwd_out)
            rev_outs.append(rev_out)
            if resample:
                out_refs.append(out_refs[-1].q_out.q_out.q1_out)
            else:
                out_refs.append(out_refs[-1].q_out.q1_out)

            # Check weights, proposal_trace, and, target_trace at every step
            assert (valeq(proposal_trace_manual, stats_nvi_dec["proposal_trace"][-(k+1)])), "proposal_traces not equal {}".format(k)
            assert (valeq(target_trace_manual, stats_nvi_dec["target_trace"][-(k+1)])), "proposal_traces not equal {}".format(k)
            assert (lw_ == stats_nvi_dec["lw"][-(k+1)]).all(), "log weights not equal {}".format(k)

def test_nvi_sampling_scheme(num_seeds=100):
    print(f"Testing NVI sampling scheme without resampling (num_seeds: {num_seeds})")
    test_K_step_nvi(8, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=True, num_seeds=num_seeds)
    print(f"Testing NVI sampling scheme with resampling (num_seeds: {num_seeds})")
    test_K_step_nvi(8, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False, num_seeds=num_seeds)

def plot_sample_hist(ax, samples, sort=True, bins=50, range=None, weight_cm=False, **kwargs):
    import numpy as np
    # ax.tick_params(bottom=False, top=False, left=False, right=False,
    #                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    # ax.grid(False)
    x, y = samples.detach().cpu().numpy().T
    ax.scatter(x, y)
    return
    mz, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True, range=range)
    X, Y = np.meshgrid(x_e, y_e)
    if weight_cm:
        raise NotImplemented()
    else:
        ax.imshow(mz, **kwargs)

def plot_nvi_run(targets, forwards, reverses, losses, ess, lZ_hat, batch_dim, sample_dims, suffix=""):
    fig = plt.figure(figsize=(K*4, 3*4))
    for k in range(K):
        ax1 = fig.add_subplot(4, K, k+1)
        ax1.plot(losses[:, k].detach().cpu().squeeze())
        if k == 0:
            ax1.set_ylabel("loss", fontsize=18)
        ax2 = fig.add_subplot(4, K, k+1+K)
        ax2.plot(ess[:, k].detach().cpu().squeeze())
        if k == 0:
            ax2.set_ylabel("ess", fontsize=18)
        ax3 = fig.add_subplot(4, K, k+1+2*K)
        ax3.plot(lZ_hat[:, k].detach().cpu().squeeze())
        if k == 0:
            ax3.set_ylabel("log_Z_hat", fontsize=18)

    tr = forward_sample(targets[0], forwards, sample_shape=(10000,), batch_dim=batch_dim, sample_dims=sample_dims,)
    samples = [(t.name, tr[t.name].value) for t in targets[1:]]  # skip the initial gaussian proposal

    for k in range(K):
        ax4 = fig.add_subplot(4, K, k+1+3*K)
        label, X = samples[k]
        plot_sample_hist(ax4, X)
        ax4.set_xlabel(label, fontsize=18)
        if k == 0:
            ax4.set_ylabel("samples", fontsize=18)

    fig.tight_layout(pad=1.0)
    fig.savefig("results/nvi_run_{}.pdf".format(suffix), bbox_inches='tight')

def forward_sample(proposal, kernels, sample_shape=(2000,), sample_dims=None, batch_dim=None):
    q = proposal
    for k in kernels:
        q = Compose(q, k)
    out = q(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
    return out.trace

def test_nvi_grads(K, sample_shape=(11,), batch_dim=1, sample_dims=0, resample=False, iterations=100, loss_fn=nvo_avo):
    out = paper_model(K+1)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]
    optimizer = adam([*targets, *forwards, *reverses])

    # Check inizialization
    test_initial_weights = forwards[0].net.map_cov[0].weight
    assert (test_initial_weights == 0.).all()

    losses= []
    lws = []
    tqdm_iterations = trange(iterations)
    for i in tqdm_iterations:
        out = nvi_declarative(targets, forwards, reverses, loss_fn,
                              sample_shape, batch_dim=batch_dim, sample_dims=sample_dims,
                              resample=resample)
        loss = out.loss.mean()
        lw = out.q_out.log_weight if out.type == "Resample" else out.log_weight
        ess = effective_sample_size(lw, sample_dims=sample_dims)
        lZ_hat = log_Z_hat(lw, sample_dims=sample_dims)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # =================================================== #
        #                    tqdm updates                     #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        tqdm_iterations.set_postfix(loss="{:09.4f}".format(loss.item()),
                                    ess="{:09.4f}".format(ess.item()),
                                    log_Z_hat="{:09.4f}".format(lZ_hat.item()))
        # =================================================== #

        lw, loss, proposal_traces, target_traces = _get_stats(out, resample=resample)
        losses.append(torch.stack(loss, dim=0))
        lws.append(torch.stack(lw, dim=0))

    lws = torch.stack(lws, dim=0)
    losses = torch.stack(losses, dim=0)
    ess = effective_sample_size(lws, sample_dims=2)
    lZ_hat = effective_sample_size(lws, sample_dims=2)
    # SHould only work if we detach first loss term
    assert (forwards[0].net.map_cov[0].weight == 0.).all()
    assert (reverses[0].net.map_cov[0].weight == 0.).all()
    assert (forwards[0].net.map_mu[0].weight == 0.).all()
    assert (reverses[0].net.map_mu[0].weight == 0.).all()
    assert (forwards[0].net.map_mu[0].bias == 0.).all()
    assert (reverses[0].net.map_mu[0].bias == 0.).all()

    plot_nvi_run(targets, forwards, reverses, losses, ess, lZ_hat, batch_dim=batch_dim, sample_dims=sample_dims,
                 suffix="_{}_{}_{}_{}".format(K, iterations, loss_fn.__name__, "_r" if resample else ""))
    # plt.show()

if __name__ == '__main__':
    S = 288
    K = 2
    iterations = 10
    objective = nvo_rkl
    # objective = nvo_avo
    # test_K_step_nvi(K, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False, num_seeds=10)
    # test_K_step_nvi(K, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=True, num_seeds=10)
    test_nvi_grads(K, sample_shape=(S//K,1), iterations=iterations, batch_dim=1, sample_dims=0, resample=False, loss_fn=objective)
    print("done!")
