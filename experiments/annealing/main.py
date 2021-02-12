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
from combinators.utils import adam
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
from combinators.objectives import nvo_rkl, nvo_avo, mb0, mb1, _estimate_mc, eval_nrep
from combinators.inference import *
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat

import experiments.visualize as V
from experiments.annealing.models import mk_model, sample_along, paper_model

def _log_weights(out, ret=[])->[Tensor]:
    if out.type == "Propose":
        _ret = ret + ([out.log_weight.detach().cpu()])
        return _log_weights(out.q.q2, _ret)
    else:
        _ret = ret + ([out.log_weight.detach().cpu()])
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
    print(loss)
    return loss

def nvi_declarative(targets, forwards, reverses, sample_shape, batch_dim, sample_dims):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(p=Extend(p, rev),
                    q=Compose(fwd, q),
                    loss_fn=print_and_sum_loss, _debug=True, ix=k, loss0=torch.zeros(1, **kw_autodevice()))

    out = q(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)

    return _log_weights(out), out.loss, out

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

if __name__ == '__main__':
    batch_dim=0
    sample_dims=1
    sample_shape = (3, 100, 2)
    out = paper_model()
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]
    nvi_declarative(targets, forwards, reverses, sample_shape, batch_dim=batch_dim, sample_dims=sample_dims)
    # main(nvi_eager_resample, K=8, seed=1, eval_break=50, num_iterations=10000)
