#!/usr/bin/env python3


import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
import math

from typing import Tuple

from combinators.inference import _dispatch
from combinators import Forward, Reverse, Propose, Condition, Resample, RequiresGrad
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.tensor.utils import autodevice, kw_autodevice
import combinators.debug as debug
from experiments.apgs_gmm.models import mk_target, Enc_rws_eta, Enc_apg_z, Enc_apg_eta, Generative, GenerativeOriginal
from experiments.apgs_gmm.objectives import resample_variables

if debug.runtime() == 'jupyter':
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm

def oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x):
    _q_eta_z_out = enc_rws_eta(x, prior_ng=generative.prior_ng)
    _q_eta_z_out = enc_apg_z(_q_eta_z_out.trace, _q_eta_z_out.output, x=x)
    log_q = _q_eta_z_out.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)

    p = og.x_forward(_q_eta_z_out.trace, x)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0)
    _loss = (w * (- log_q)).sum(0).mean()
    return _loss, log_w, _q_eta_z_out

def rws_objective_eager(enc_rws_eta, enc_apg_z, generative, og, x, compare=False, sample_size=None):
    """ One-shot for eta and z, like a normal RWS """
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp  = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    out = prp(x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.log_joint)).sum(0).mean()

    return loss, mk_metrics(loss, w, out)

def apg_objective(enc_rws_eta, enc_apg_z, generative, og, x, num_sweeps, compare=True, sample_size=None):
    """
    Amortized Population Gibbs objective in GMM problem
    ==========
    abbreviations:
    K -- number of clusters
    D -- data dimensions (D=2 in GMM)
    S -- sample size
    B -- batch size
    N -- number of data points in one (GMM) dataset
    ==========
    variables:
    ob  :  S * B * N * D, observations, as data points
    tau :  S * B * K * D, cluster precisions, as global variables
    mu  :  S * B * K * D, cluster means, as global variables
    eta : ={tau, mu} global block
    z   :  S * B * N * K, cluster assignments, as local variables
    ==========
    """
    #(enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)

        from combinators.resampling.strategies import APGResampler
        resampler = APGResampler('systematic', sample_size, False, 'cpu')
        q_eta_z = resample_variables(resampler, _q_eta_z_out.trace, log_weights=_log_w)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp  = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    out = prp(x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.weights)).sum(0).mean()

    for m in range(num_sweeps-1):
            log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x, metrics, result_flags)
            q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_eta)
            log_w_z, q_eta_z, metrics = apg_update_z(enc_apg_z, generative, q_eta_z, x, metrics, result_flags)
            q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_z)
    if result_flags['loss_required']:
        metrics['loss'] = torch.cat(metrics['loss'], 0)
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0)
    if result_flags['mode_required']:
        metrics['E_tau'] = torch.cat(metrics['E_tau'], 0)
        metrics['E_mu'] = torch.cat(metrics['E_mu'], 0)
        metrics['E_z'] = torch.cat(metrics['E_z'], 0)  # (num_sweeps) * B * N * K
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0)  # (num_sweeps) * S * B
    return metrics

def mk_metrics(loss, w, out, num_sweeps=1, ess_required=True, mode_required=True, density_required=True, mode_required_with_z=False):
    with torch.no_grad():
        metrics = dict()
        metrics['loss'] = loss.mean().cpu().item()
        if ess_required:
            metrics['ess'] = (1. / (w**2).sum(0)).mean().cpu().item()
        if mode_required:
            q_eta_z = out.proposal.trace
            metrics['E_tau'] = (q_eta_z['precisions'].dist.concentration / q_eta_z['precisions'].dist.rate).mean(0).detach().mean().cpu().item()
            metrics['E_mu'] = q_eta_z['means'].dist.loc.mean(0).detach().mean().cpu().item()
            if mode_required_with_z:
                # this is stable at 1/3
                metrics['E_z'] = q_eta_z['states'].dist.probs.mean(0).detach().mean().cpu().item()
        if density_required:
            metrics['density'] = out.target.log_joint.detach().mean().cpu().item()

        if num_sweeps > 1:
            pass
            # exc_kl, inc_kl= kls_eta(models, x, z_true)
            # if 'inc_kl' in metrics:
            #     metrics['inc_kl'] += inc_kl
            # else:
            #     metrics['inc_kl'] = inc_kl
            # if 'exc_kl' in metrics:
            #     metrics['exc_kl'] += exc_kl
            # else:
            #     metrics['exc_kl'] = exc_kl
        return metrics

def train(objective, models, target, og, data, assignments, num_epochs, sample_size, batch_size, normal_gamma_priors, with_tensorboard=False, lr=1e-3, seed=1, eval_break=50, is_smoketest=False):
    # (num_clusters, data_dim, normal_gamma_priors, num_iterations=10000, num_samples = 600batches=50)
    """ data size  S * B * N * 2 """
    # Setup
    debug.seed(seed)
    writer = debug.MaybeWriter(enable=with_tensorboard)
    loss_ct, loss_sum = 0, 0.0

    [enc_rws_eta, enc_apg_z, enc_apg_eta], generative = models, target

    assert all([len(list(k.parameters())) >  0 for k in models])
    optimizer = optim.Adam([dict(params=x.parameters()) for x in models], lr=lr)
    num_batches = int((data.shape[0] / batch_size))
    # sample_shape = (batch_size, sample_size)
    epochs = range(1) if is_smoketest else trange(num_epochs, desc='Epochs', position=1)
    for e in epochs:
        data, assignments = shuffler(data, assignments)

        num_batches = 3 if is_smoketest else num_batches
        with trange(num_batches, desc=f'Batch {{:{int(math.log(num_epochs, 10))}d}}'.format(e+1), position=0) as batches:
            for b in batches:
                optimizer.zero_grad()
                x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)
                loss, metrics = objective(enc_rws_eta, enc_apg_z, generative, og, x, sample_size)

                loss.backward()
                optimizer.step()

                # REPORTING
                # ---------------------------------------
                with torch.no_grad():
                    batches.set_postfix_str(";".join([f'{k}={{: .2f}}'.format(v) for k, v in metrics.items()])) # loss=loss.detach().cpu().mean().item(), **metrics)

def shuffler(data, assignments):
    """
    shuffle the GMM datasets by both permuting the order of GMM instances (w.r.t. DIM1) and permuting the order of data points in each instance (w.r.t. DIM2)
    """
    concat_var = torch.cat((data, assignments), dim=-1)
    DIM1, DIM2, DIM3 = concat_var.shape
    indices_DIM1 = torch.randperm(DIM1)
    concat_var = concat_var[indices_DIM1]
    indices_DIM2 = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)], dim=0)
    concat_var = torch.gather(concat_var, 1, indices_DIM2.unsqueeze(-1).repeat(1, 1, DIM3))
    return concat_var[:,:,:2], concat_var[:,:,2:]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('apgs_gmm')
    parser.add_argument('--simulate', default=False, type=bool)
    parser.add_argument('--num_sweeps', default=1, type=int)
    args = parser.parse_args()

    data_path='../../data/gmm/'
    if args.simulate:
        from experiments.apgs_gmm.simulator import Sim_GMM
        simulator = Sim_GMM(N=60, K=3, D=2, alpha=2.0, beta=2.0, mu=0.0, nu=0.1)
        simulator.sim_save_data(num_seqs=10000, data_path=data_path)

    data = torch.from_numpy(np.load(f'{data_path}ob.npy')).float()
    assignments = torch.from_numpy(np.load(f'{data_path}assignment.npy')).float()

    # hyperparameters
    num_epochs=100
    batch_size=10
    budget=100
    num_sweeps=1
    lr=2e-4
    num_clusters=K=3
    data_dim=D=2
    num_hidden=64
    is_smoketest=debug.is_smoketest()

    normal_gamma_priors = dict(
        mu=torch.zeros((num_clusters, data_dim)),
        nu=torch.ones((num_clusters, data_dim)) * 0.1,
        alpha=torch.ones((num_clusters, data_dim)) * 2.0,
        beta=torch.ones((num_clusters, data_dim)) * 2.0,
    )
    # computable params
    num_batches = int((data.shape[0] / batch_size))
    sample_size = int(budget / num_sweeps)

    # Models
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden=30)
    enc_apg_eta = Enc_apg_eta(K, D)
    generative = Generative(K, normal_gamma_priors)
    og = GenerativeOriginal(K, D, False, 'cpu')
    if args.num_sweeps == 1: ## rws method
        train(
            objective=rws_objective_eager,
            models=[enc_rws_eta, enc_apg_z, enc_apg_eta],
            target=generative,
            og=og,
            data=data,
            assignments=assignments,
            normal_gamma_priors=normal_gamma_priors,
            num_epochs=num_epochs,
            batch_size=batch_size,
            sample_size=sample_size,
            is_smoketest=is_smoketest,
        )
    else: ## apgs sampler
        train(
            objective=apg_objective,
            models=[enc_rws_eta, enc_apg_z, enc_apg_eta],
            target=generative,
            og=og,
            data=data,
            assignments=assignments,
            normal_gamma_priors=normal_gamma_priors,
            num_epochs=num_epochs,
            batch_size=batch_size,
            sample_size=sample_size,
            is_smoketest=is_smoketest,
        )
    print('tada!')
