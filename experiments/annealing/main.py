#!/usr/bin/env python3
import torch
import math
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt

import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
from combinators.densities.kernels import MultivariateNormalKernel, MultivariateNormalLinearKernel, NormalLinearKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl, nvo_avo, mb0, mb1, _estimate_mc, eval_nrep
from combinators import Forward, Reverse, Propose
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat

import experiments.annealing.visualize as V
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


def nvi(targets, forwards, reverses, sample_shape):
    q0 = targets[0]
    p_prv_tr, out0 = q0(sample_shape=sample_shape)

    loss = torch.zeros(1, **kw_autodevice())
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []
    lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []
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
        # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
        loss += nvo_avo(lv)

        lvss.append(lv)
    return lw, lvss, loss

def main(K=8, seed=1, eval_break=50, num_iterations=10000):
    # Setup
    torch.manual_seed(seed)
    num_samples = 256
    sample_shape=(num_samples,)

    # Models
    out = paper_model()
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    optimizer = optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-3)

    with trange(num_iterations) as bar:
        for i in bar:
            lw, lvss, loss = nvi(targets, forwards, reverses, sample_shape)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            # ---------------------------------------
            with torch.no_grad():
                lvs = torch.stack(lvss, dim=0)
                lws = torch.cumsum(lvs, dim=1)
                ess = effective_sample_size(lws, sample_dims=-1)
                lzh = log_Z_hat(lws, sample_dims=-1)

                loss_scalar = loss.detach().cpu().mean().item()

                report(writer, ess, lzh, loss_scalar, i, eval_break, targets, forwards)

                loss_ct += 1
                loss_sum += loss_scalar
                # Update progress bar
                if i % 10 == 0:
                    loss_avg = loss_sum / loss_ct
                    loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
                    logZh_template = 'logZhat[-1]={:.4f}'.format(lzh[-1].cpu().item())
                    ess_template = 'ess[-1]={:.4f}'.format(ess[-1].cpu().item())
                    loss_ct, loss_sum  = 0, 0.0
                    bar.set_postfix_str("; ".join([loss_template, ess_template, logZh_template]))


if __name__ == '__main__':
    main(K=8, seed=1, eval_break=50, num_iterations=10000)
