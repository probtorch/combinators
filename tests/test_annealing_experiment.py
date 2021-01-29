#!/usr/bin/env python3
import torch
import os
import math
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt
from pytest import mark, fixture

import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
from combinators.densities.kernels import MultivariateNormalKernel, MultivariateNormalLinearKernel, NormalLinearKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl, nvo_avo, mb0, mb1, _estimate_mc, eval_nrep
from combinators import Forward, Reverse, Propose, Condition, RequiresGrad, Resample
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat
from tests.utils import is_smoketest, seed
import combinators.debug as debug

import experiments.visualize as V

from experiments.annealing.models import mk_model, sample_along, paper_model

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


def experiment_runner(is_smoketest, trainer):
    debug.seed()
    eval_break=50
    num_iterations=3 if is_smoketest else 1000
    # Setup
    # num_samples = 5 if is_smoketest else 256
    num_samples=5
    sample_shape=(num_samples,)

    # Models
    out = paper_model()
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    # logging
    writer = debug.MaybeWriter(is_smoketest)
    loss_ct, loss_sum, loss_moving_100 = 0, 0.0, []

    with torch.no_grad():
        lvss, loss = trainer(-1, targets, forwards, reverses, sample_shape)
        loss_ct, loss_sum, ess, lzh, loss_scalar = bar_report(-1, lvss, loss, loss_ct, loss_sum)
        print('===========================================================')
        print('loss0={:.4f}, ess_mean0={:.4f}, lzh0={}'.format(loss_scalar, ess.mean().cpu().item(), lzh))

    optimizer = optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-3)

    with trange(num_iterations) as bar:
        for i in bar:

            lvss, loss = trainer(i, targets, forwards, reverses, sample_shape)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                loss_ct, loss_sum, ess, lzh, loss_scalar = bar_report(i, lvss, loss, loss_ct, loss_sum)
                report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards)
                loss_moving_100.append(loss_scalar)
                if len(loss_moving_100) > 100:
                    loss_moving_100.pop(0)

                bar.set_postfix_str("; ".join([
                    'loss={: .4f}'.format(loss_scalar),
                    'logZhat[-1]={:.4f}'.format(lzh[-1].cpu().item()),
                    'ess[-1]={:.4f}'.format(ess[-1].cpu().item())
                ]))
    print('loss={:.4f}, ess_mean={:.4f}, lzh={}'.format(sum(loss_moving_100)/100, ess.mean().cpu().item(), lzh))
    print('-----------------------------------------------------------')

def bar_report(i, lvss, loss, loss_ct, loss_sum):
    # REPORTING
    # ---------------------------------------
    with torch.no_grad():
        lvs = torch.stack(lvss, dim=0)
        lws = torch.cumsum(lvs, dim=1)
        ess = effective_sample_size(lws, sample_dims=-1)
        lzh = log_Z_hat(lws, sample_dims=-1)

        loss_scalar = loss.detach().cpu().mean().item()


        loss_ct += 1
        loss_sum += loss_scalar
        # Update progress bar
        if i % 10 == 0:
            loss_avg = loss_sum / loss_ct
            loss_ct, loss_sum  = 0, 0.0
        return loss_ct, loss_sum, ess, lzh, loss_scalar



def nvi_eager(i, targets, forwards, reverses, sample_shape):
    q0 = targets[0]
    p_prv_tr, _, _ = q0(sample_shape=sample_shape)

    loss = torch.zeros(1, **kw_autodevice())
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []

    for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
        q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO), _step=k)
        p_ext = Reverse(p, rev, _step=k)
        extend = Propose(target=p_ext, proposal=q_ext, _step=k)
        state = extend(sample_shape=sample_shape, sample_dims=0)
        lv = state.log_weight

        p_prv_tr = state.trace

        lw += lv

        # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
        objective_loss = nvo_avo(lv)
        loss += objective_loss

        lvss.append(lv)

    return lvss, loss

def nvi_eager_resample(i, targets, forwards, reverses, sample_shape):
    q0 = targets[0]
    p_prv_tr, _, _ = q0(sample_shape=sample_shape)

    loss = torch.zeros(1, **kw_autodevice())
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []

    for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
        q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO), _step=k)
        p_ext = Reverse(p, rev, _step=k)
        extend = Resample(Propose(target=p_ext, proposal=q_ext, _step=k))

        state = extend(sample_shape=sample_shape, sample_dims=0)
        lv = state.log_weight

        p_prv_tr = state.trace

        lw += lv

        # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
        objective_loss = nvo_avo(lv)
        loss += objective_loss

        lvss.append(lv)

    return lvss, loss

def test_annealing_eager(is_smoketest):
    print("test_annealing_eager")
    experiment_runner(is_smoketest, nvi_eager)

def test_annealing_eager_resample(is_smoketest):
    print("test_annealing_eager_resample")
    experiment_runner(is_smoketest, nvi_eager_resample)

def _log_joint(out, ret=[])->[Tensor]:
    _ret = ret + ([out.log_joint.detach().cpu()] if out.log_joint is not None else [])
    if 'proposal' not in out:
        return _ret
    else:
        return _log_joint(out.proposal.program, _ret)


def print_and_sum_loss(lv, loss):
    out = loss + nvo_avo(lv)
    print(out)
    return out


def nvi_declarative(i, targets, forwards, reverses, sample_shape):
    def mk_step(q, p, fwd, rev, k)->Propose:
        q_ext = Forward(fwd, q)
        p_ext = Reverse(p, rev)
        return Propose(target=p_ext, proposal=q_ext, loss_fn=print_and_sum_loss, _debug=True, _step=k, loss0=torch.zeros(1, **kw_autodevice()))

    proposal = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        proposal = mk_step(proposal, p, fwd, rev, k)

    out = proposal(sample_shape=sample_shape, sample_dims=0)

    return _log_joint(out), out.loss

@mark.skip("accumulation of gradient needs to be fixed")
def test_annealing_declarative():
    is_smoketest = True
    print("test_annealing_declarative")
    experiment_runner(is_smoketest, nvi_declarative)
