#!/usr/bin/env python3
from combinators.inference import Condition, Resample
import torch
import math
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt

import combinators.trace.utils as trace_utils
from combinators.utils import adam, ppr
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal, Density
from combinators.objectives import nvo_rkl, nvo_avo, mb0, mb1, _estimate_mc, eval_nrep
from combinators.inference import *
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.resampling.strategies import Systematic

import experiments.visualize as V
from experiments.annealing.models import mk_model, sample_along, paper_model

def _log_weights(out, ret=[])->[Tensor]:
    _ret = [out.log_weight.detach().cpu()] + ret

    if out.q_out.q1_out.type == "Propose":
        use = out.q_out.q1_out
        return _log_weights(use, _ret)
    elif out.q_out.q2_out.type == "Propose":
        use = out.q_out.q2_out
        return _log_weights(use, _ret)
    else:
        return _ret

def report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards):
    with torch.no_grad():
        # loss
        writer.add_scalar('loss', loss_scalar, i)

        # ESS
        for step, x in zip(range(1,len(ess)+1), ess):
            writer.add_scalar(f'ess/step-{step}', x, i)

        # logZhat
        for step, x in zip(range(1,len(lzh)+1), lzh):
            writer.add_scalar(f'log_Z_hat/step-{step}', x, i)

        # show samples
        if i % eval_break == 0:
            samples = sample_along(targets[0], forwards)
            fig = V.scatter_along(samples)
            writer.add_figure('overview', fig, global_step=i, close=True)

def bar_report(i, lvss, loss):
    with torch.no_grad():
        lvs = torch.stack(lvss, dim=0)
        lws = torch.cumsum(lvs, dim=1)
        ess = effective_sample_size(lws, sample_dims=-1)
        lzh = log_Z_hat(lws, sample_dims=-1)

        loss_scalar = loss.detach().cpu().mean().item()

        return ess, lzh, loss_scalar

def print_and_sum_loss(out, loss):
    loss = loss + nvo_avo(out.lv)
    # print(loss)
    return loss

def nvi_declarative(targets, forwards, reverses, sample_shape, batch_dim, sample_dims, resample=False):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(p=Extend(p, rev),
                    q=Compose(fwd, q),
                    loss_fn=print_and_sum_loss, _debug=True, ix=k, loss0=torch.zeros(1, **kw_autodevice()))
        q = Resample(q)
    out = q(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
    return out

def main(trainer, K=8, seed=1, eval_break=50, num_iterations=10000, num_samples = 256):
    # Setup
    debug.seed(seed)
    sample_shape=(num_samples,)

    # Models
    out = paper_model()
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    # logging
    writer = SummaryWriter()

    optimizer = adam([*forwards, *reverses], lr=1e-3)

    with trange(num_iterations) as bar:
        for i in bar:
            lvss, loss = trainer(i, targets, forwards, reverses, sample_shape)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                ess, lzh, loss_scalar = bar_report(i, lvss, loss)
                report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards)

                bar.set_postfix_str("; ".join([
                    'loss={: .4f}'.format(loss_scalar),
                    'logZhat[-1]={:.4f}'.format(lzh[-1].cpu().item()),
                    'ess[-1]={:.4f}'.format(ess[-1].cpu().item())
                ]))

def compute_nvi_weight(proposal, target, forward, reverse, lw_in=0., sample_shape=(10, 5), batch_dim=0, sample_dims=1, resample=False):
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

def test_1_step_nvi(sample_shape=(10, 5), batch_dim=0, sample_dims=1):
    out = mk_model(2)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    seeds = torch.arange(100)
    for seed in seeds:
        debug.seed(seed)
        out = nvi_declarative(targets, forwards, reverses, sample_shape, batch_dim=batch_dim, sample_dims=sample_dims)

        debug.seed(seed)
        lw, _ = compute_nvi_weight(targets[0], targets[1], forwards[0], reverses[0], lw_in=0.,
                                   sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
        assert (lw == out.log_weight).all()

    # main(nvi_eager_resample, K=8, seed=1, eval_break=50, num_iterations=10000)

def test_K_step_nvi(K, sample_shape=(10, 5), batch_dim=0, sample_dims=1, resample=False):
    out = mk_model(K+1)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    seeds = torch.arange(10)
    for seed in seeds:
        debug.seed(seed)
        out = nvi_declarative(targets, forwards, reverses,
                              sample_shape, batch_dim=batch_dim, sample_dims=sample_dims,
                              resample=resample)

        debug.seed(seed)
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

            lws.append(lw)
            lws_.append(lw_)
            cond_traces.append(cond_trace)
            target_outs.append(target_out)
            proposal_outs.append(proposal_out)
            fwd_outs.append(fwd_out)
            rev_outs.append(rev_out)
            out_refs.append(out_refs[-1].q_out.q_out.q1_out)
        assert (lw == out.log_weight).all()


if __name__ == '__main__':
    test_K_step_nvi(8, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=True)


# check proposal log_probs
#   proposal_outs[-1].trace['g1'].log_prob
#   out.q_out.q_out.trace['g1'].log_prob
# check fwd log_probs
#   out.q_out.q_out.trace['g2'].log_prob
#   fwd_outs[-1].trace['g2'].log_prob
# check target log_probs
#   out.q_out.p_out.trace['g2'].log_prob
#   target_outs[-1].trace['g2'].log_prob
# check rev log_probs
#   out.q_out.p_out.trace_star['g1'].log_prob
#   rev_outs[-1].trace['g1'].log_prob
# --> log_prob are all good individually but weight is not
# => Check individual parts of the weight now
# lw_1 - incomming log weight (after resampling) of last step
#   out.q_out.q_out.log_weight
#   lws[0]
# lw_2 - log weight of extend in last propose
#   out.q_out.p_out.log_weight
#   target_outs[-1].trace['g2'].log_prob + rev_outs[-1].trace['g1'].log_prob
# lu - log weight for reused variables under the proposal
#   fwd_outs[-1].trace['g2'].log_prob + proposal_outs[-1].trace['g1'].log_prob
#   out.q_out.lu
# -> This is not correct -> lu_star somehow double counts

## Old story -> fixed!!!
# Compare fwd_kernels before faleure
#   out_refs[-3].q_out.q_out.q2_out.trace['g2'].dist.loc[:, 0]
#   fwd_outs[-1].trace['g2'].dist.loc[:, 0]
# -> forward kernels should use inputs (both seem to be resampled):
#   out_refs[-2].trace['g1'].value[:, 0]
#   cond_traces[1]['g1'].value[:, 0]
# -> However, combinator kernel uses
#   out_refs[-2].q_out.trace['g1'].value[:, 0]
# -> This is the input before it was resampled!!!

