#!/usr/bin/env python3
import torch
from torch import Tensor
from tqdm import trange
from typing import List
from matplotlib import pyplot as plt
from logging import getLogger

from combinators import adam, autodevice
from combinators import Propose, Extend, Compose, Resample
from combinators import effective_sample_size, log_Z_hat
from combinators.utils import save_models, load_models, models_as_dict

from experiments.annealing.models import paper_model
from experiments.annealing.objectives import nvo_rkl, nvo_avo, stl_trace

logger = getLogger(__file__)

def traverse_proposals(fn, out, memo=[])->List[Tensor]:
    if out.type == Propose:
        return traverse_proposals(fn, out.q_out, [fn(out)] + memo)
    elif out.type == Extend:
        raise ValueError("impossible! traverse proposal will never arrive here")
    elif out.type == Compose:
        valid_infs = [Compose, Propose, Resample]
        q1_is_inf = out.q1_out.type in valid_infs
        q2_is_inf = out.q2_out.type in valid_infs
        if not (q1_is_inf ^ q2_is_inf):
            return memo
        else:
            return traverse_proposals(fn, out.q1_out if q1_is_inf else out.q2_out, memo)
    elif out.type == Resample:
        return traverse_proposals(fn, out.q_out, memo)
    elif out.type == Condition:
        raise ValueError("impossible! traverse proposal will never arrive here")
    else:
        return memo

def get_stats(out, detach=True):
    fn = lambda out: (out.log_weight.detach(), out.loss.detach(), out.proposal_trace, out.target_trace, out)
    ret = traverse_proposals(fn, out)
    lws, losses, proposal_trace, target_trace, outs = zip(*ret)
    return dict(lw=lws, loss=losses, proposal_trace=proposal_trace, target_trace=target_trace, out=outs)

class LossFn(object):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @property
    def __name__(self):
        return self.loss_fn.__name__

    def __call__(self, out, loss):
        return loss + self.loss_fn(out)

def nvi_declarative(targets, forwards, reverses, loss_fn, resample=False):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(p=Extend(p, rev),
                    q=Compose(fwd, q, _debug=True),
                    loss_fn=LossFn(loss_fn),
                    ix=k,
                    _no_reruns=False,
                    _debug=True,
                    transf_q_trace=None if loss_fn.__name__ == "nvo_avo" else stl_trace,
                    )
        if resample and k < len(forwards) - 1:
            q = Resample(q)
    return q

def plot_sample_hist(ax, samples, sort=True, bins=50, range=None, weight_cm=False, **kwargs):
    import numpy as np
    ax.grid(False)
    samples = torch.flatten(samples[:, :10], end_dim=-2)
    x, y = samples.detach().cpu().numpy().T
    mz, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True, range=range)
    X, Y = np.meshgrid(x_e, y_e)
    if weight_cm:
        raise NotImplemented()
    else:
        ax.imshow(mz, **kwargs)

def plot(losses, ess, lZ_hat, samples, filename=None):
    K = losses.shape[1]
    fig = plt.figure(figsize=(K*4, 3*4), dpi=300)
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

    for k in range(K):
        ax4 = fig.add_subplot(4, K, k+1+3*K)
        label, X = samples[k]
        plot_sample_hist(ax4, X, bins=150)
        ax4.set_xlabel(label, fontsize=18)
        if k == 0:
            ax4.set_ylabel("samples", fontsize=18)

    fig.tight_layout(pad=1.0)
    if filename is not None:
        fig.savefig("figures/{}".format(filename), bbox_inches='tight')

def check_weights_zero(forwards, reverses):
    for forward, reverse in zip(forwards, reverses):
        assert (forward.net.map_cov[0].weight == 0.).all()
        assert (reverse.net.map_cov[0].weight == 0.).all()
        assert (forward.net.map_mu[0].weight == 0.).all()
        assert (reverse.net.map_mu[0].weight == 0.).all()
        assert (forward.net.map_mu[0].bias == 0.).all()
        assert (reverse.net.map_mu[0].bias == 0.).all()

def nvi_test(nvi_program, sample_shape, batch_dim=1, sample_dims=0):
    out = nvi_program(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim, _debug=True)
    stats_nvi_run = get_stats(out)
    lw = torch.stack(stats_nvi_run['lw'])
    loss = torch.stack(stats_nvi_run['loss'])
    proposal_traces = stats_nvi_run['proposal_trace']

    ess = effective_sample_size(lw, sample_dims=sample_dims+1)
    lZ_hat = log_Z_hat(lw, sample_dims=sample_dims+1)
    samples = [('g{}'.format(t+1), proposal_traces[t]['g{}'.format(t+1)].value) for t in range(len(proposal_traces))]  # skip the initial gaussian proposal
    return loss, ess, lZ_hat, samples, out


def nvi_train(q, targets, forwards, reverses,
          sample_shape=(11,),
          batch_dim=1,
          sample_dims=0,
          iterations=100,):

    # get shapes for metrics aggregation
    stats_nvi_run = get_stats(q(None, sample_shape=sample_shape, batch_dim=batch_dim, sample_dims=sample_dims, _debug=True))
    mk_metric = lambda kw: torch.zeros(iterations,  len(stats_nvi_run[kw]), *stats_nvi_run[kw][0].shape)
    full_losses, full_lws = mk_metric('loss'), mk_metric('lw')

    # Check inizialization
    optimizer = adam([*targets, *forwards, *reverses])
    check_weights_zero(forwards[:1], reverses[:1])

    tqdm_iterations = trange(iterations)
    for i in tqdm_iterations:
        out = q(None, sample_shape=sample_shape, batch_dim=batch_dim, sample_dims=sample_dims, _debug=True)
        loss = out.loss.mean()
        lw = out.q_out.log_weight if out.type == "Resample" else out.log_weight
        ess = effective_sample_size(lw, sample_dims=sample_dims)
        lZ_hat = log_Z_hat(lw, sample_dims=sample_dims)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =================================================== #
        #                    tqdm updates                     #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        tqdm_iterations.set_postfix(loss="{:09.4f}".format(loss.item()),
                                    ess="{:09.4f}".format(ess.item()),
                                    log_Z_hat="{:09.4f}".format(lZ_hat.item()))
        # =================================================== #
        stats_nvi_run = get_stats(out)
        lw, loss, _, _ = stats_nvi_run['lw'], stats_nvi_run['loss'], stats_nvi_run['proposal_trace'], stats_nvi_run['target_trace']
        torch.stack(loss, dim=0, out=full_losses[i])
        torch.stack(lw, dim=0, out=full_lws[i])

    ess = effective_sample_size(full_lws, sample_dims=2)
    lZ_hat = log_Z_hat(full_lws, sample_dims=2)
    return q, full_losses, ess, lZ_hat

def save_nvi_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    save_models(
        models_as_dict(
            [targets, forwards, reverses],
            ["targets", "forwards", "reverses"]
        ),
        filename="{}.pt".format(filename), weights_dir="./weights")

def load_nvi_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    load_models(models_as_dict([targets, forwards, reverses], ["targets", "forwards", "reverses"]), filename="./{}.pt".format(filename), weights_dir="./weights")

def mk_model(K, optimize_path=False):
    mod = paper_model(K, optimize_path=optimize_path)
    targets, forwards, reverses = [[m.to(autodevice()) for m in mod[n]] for n in ['targets', 'forwards', 'reverses']]
    return targets, forwards, reverses

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser('Combinators annealing stats')
    # data config
    parser.add_argument('--objective', default='nvo_rkl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resample', default=False, type=bool)
    parser.add_argument('--iterations', default=20000, type=int)
    parser.add_argument('--num_targets', default=4, type=int)
    parser.add_argument('--sample_budget', default=288, type=int)
    parser.add_argument('--optimize_path', default=False, type=bool)

    # CI config
    parser.add_argument('--smoketest', default=False, type=bool)

    args = parser.parse_args()

    S = args.sample_budget
    K = args.num_targets
    seed = args.seed
    resample = args.resample
    iterations = args.iterations
    optimize_path = args.optimize_path

    if not os.path.exists("./weights/"):
        os.makedirs("./weights/")

    if not os.path.exists("./metrics/"):
        os.makedirs("./metrics/")

    if args.objective == "nvo_avo":
        objective = nvo_avo
    elif args.objective == "nvo_rkl":
        objective = nvo_rkl
    else:
        raise TypeError("objective is one of: {}".format(", ".join(["nvo_avo", "nvo_rkl"])))

    tt = True
    save_plots = False
    filename="nvi{}{}_{}_S{}_K{}_I{}_seed{}".format(
        "r" if resample else "",
        "s" if optimize_path else "",
        objective.__name__,
        S, K, iterations, seed)

    torch.manual_seed(seed)
    model = mk_model(K, optimize_path=optimize_path)
    q = nvi_declarative(*model,
                        objective,
                        resample=resample)
    S = 288
    iterations = args.iterations

    losses, ess, lZ_hat = torch.zeros(iterations, K-1, 1), torch.zeros(iterations, K-1, 1), torch.zeros(iterations, K-1, 1)

    if tt:
        q, losses, ess, lZ_hat = nvi_train(q,
                                    *model,
                                    sample_shape=(S//K,1),
                                    iterations=iterations,
                                    batch_dim=1,
                                    sample_dims=0,)
        save_nvi_model(*model, filename=filename)
    else:
        load_nvi_model(*model, filename=filename)

    q = nvi_declarative(*model,
                        objective,
                        resample=False)

    if not args.smoketest:
        losses_test, ess_test, lZ_hat_test, samples_test, _ = \
            nvi_test(q, (1000, 100), batch_dim=1, sample_dims=0)

        torch.save((losses_test.mean(1), ess_test.mean(1), lZ_hat_test.mean(1)),
                   './metrics/{}-metric-tuple_S{}_B{}-loss-ess-logZhat.pt'.format(
                       filename, 1000, 100))

        print("losses:", losses_test.mean(1))
        print("ess:", ess_test.mean(1))
        print("log_Z_hat", lZ_hat_test.mean(1))

        if save_plots:
            if not os.path.exists("./figures/"):
                os.makedirs("./figures/")

            plot(losses, ess, lZ_hat, samples_test, filename=filename)

    print("done!")
