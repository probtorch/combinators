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
from combinators.stochastic import Trace
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
import combinators.debug as debug
from experiments.apgs_gmm.models import *
from experiments.apgs_gmm.models import GenerativeLikelihoodV2, mk_target, Enc_rws_eta, Enc_apg_z, Enc_apg_eta, Generative, GenerativeOriginal
from experiments.apgs_gmm.objectives import resample_variables, apg_update_z

from torch.distributions.one_hot_categorical import OneHotCategorical as cat

if debug.runtime() == 'jupyter':
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm

def oneshot_hao(enc_rws_eta, enc_apg_z, og, x):
    _q_eta_z_out = enc_rws_eta(x, prior_ng=og.prior_ng)
    _q_eta_z_out = enc_apg_z(_q_eta_z_out.trace, _q_eta_z_out.output, x=x, prior_ng=og.prior_ng)
    log_q = _q_eta_z_out.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)

    p = og.x_forward(_q_eta_z_out.trace, x)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0)
    _loss = (w * (- log_q)).sum(0).mean()
    return _loss, log_w, _q_eta_z_out

def rws_objective_eager(enc_rws_eta, enc_apg_z, generative, og, x, compare=False, sample_size=None, num_sweeps=1):
    """ One-shot for eta and z, like a normal RWS """
    assert num_sweeps == 1
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    out = prp(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.log_joint)).sum(0).mean()
    if compare:
        breakpoint();

    return loss, mk_metrics(loss, w, out)

def rws_objective_eager(enc_rws_eta, enc_apg_z, generative, og, x, compare=True, sample_size=None, num_sweeps=1):
    """ One-shot for eta and z, like a normal RWS """
    assert num_sweeps == 1
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    out = prp(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.log_joint)).sum(0).mean()
    if compare:
        breakpoint();

    return loss, mk_metrics(loss, w, out)

def apg_update_eta(enc_apg_eta, generative, q_eta_z, x):
    """ Given local variable z, update global variables eta := {mu, tau}. """
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z, None, x, prior_ng=generative.prior_ng) ## forward kernel
    p_f = generative.forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_eta(q_eta_z, x, prior_ng=generative.prior_ng)
    p_b = generative.forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).mean()
        metrics['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if result_flags['mode_required']:
        E_tau = (q_eta_z_f['precisions'].dist.concentration / q_eta_z_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta_z_f['means'].dist.loc.mean(0).detach()
        metrics['E_tau'].append(E_tau.unsqueeze(0))
        metrics['E_mu'].append(E_mu.unsqueeze(0))
    if result_flags['density_required']:
        metrics['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, q_eta_z_f, metrics

def apg_update_z(enc_apg_z, generative, q_eta_z, x):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable state i.e. z
    """
    # forward
    q_eta_z_f = enc_apg_z(q_eta_z, x)
    p_f = generative.forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_z(q_eta_z, x)
    p_b = generative.forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).sum(-1).mean()
        metrics['loss'][-1] = metrics['loss'][-1] + loss.unsqueeze(0)
    if result_flags['mode_required']:
        E_z = q_eta_z_f['states'].dist.probs.mean(0).detach()
        metrics['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        metrics['density'].append(log_p_f.unsqueeze(0))
    return log_w, q_eta_z_f, metrics

def resample_variables(resampler, q, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    tau = q['precisions'].value
    tau_concentration = q['precisions'].dist.concentration
    tau_rate = q['precisions'].dist.rate
    mu = q['means'].value
    mu_loc = q['means'].dist.loc
    mu_scale = q['means'].dist.scale
    z = q['states'].value
    z_probs = q['states'].dist.probs
    tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
    tau_concentration = resampler.resample_4dims(var=tau_concentration, ancestral_index=ancestral_index)
    tau_rate = resampler.resample_4dims(var=tau_rate, ancestral_index=ancestral_index)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    mu_loc = resampler.resample_4dims(var=mu_loc, ancestral_index=ancestral_index)
    mu_scale = resampler.resample_4dims(var=mu_scale, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    z_probs = resampler.resample_4dims(var=z_probs, ancestral_index=ancestral_index)
    q_resampled = Trace()
    q_resampled.gamma(tau_concentration,
                      tau_rate,
                      value=tau,
                      name='precisions')
    q_resampled.normal(mu_loc,
                       mu_scale,
                       value=mu,
                       name='means')
    _ = q_resampled.variable(cat, probs=z_probs, value=z, name='states')
    return q_resampled

def apg_objective(enc_rws_eta, enc_apg_z, generative, og, x, num_sweeps, sample_size, compare=True):
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
    # # ##(enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    # # #if compare:
    # # #    debug.seed(1)
    # # #    _loss, _log_w, _q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, generative, og, x)
    # # #
    # # #    from combinators.resampling.strategies import APGResampler
    # # #    q_eta_z = resample_variables(resampler, _q_eta_z_out.trace, log_weights=_log_w)
    # # #    debug.seed(1)
    # # #
    # # ## otherwise, eager combinators looks like:
    # # #prp  = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og)
    # # #out = prp(x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)
    # # #
    # # #log_w = out.log_weight.detach()
    # # #w = F.softmax(log_w, 0)
    # # #loss = (w * (- out.proposal.weights)).sum(0).mean()
    og2, og2k = generative
    kwargs = dict(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=True)

    assert num_sweeps == 2
    from combinators.resampling.strategies import APGSResamplerOriginal
    resampler = APGSResamplerOriginal(sample_size)
    # ================================================================
    # sweep 1:
    debug.seed(1)
    _loss, log_w, q_eta_z_out = oneshot_hao(enc_rws_eta, enc_apg_z, og, x)
    q_eta_z = q_eta_z_out.trace
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w)
    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z, None, x=x, prior_ng=og.prior_ng) ## forward kernel
    q_eta_z_f = q_eta_z_f.trace
    p_f = og.x_forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    print(tensor_utils.show(log_w_f))

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

    # ================================================================
    # sweep 1:
    debug.seed(1)
    prp1 = Resample(
        Propose(
            proposal=Forward(enc_apg_z, enc_rws_eta),
            target=Reverse(og2, og2k)))

    out1 = prp1(**kwargs)
    trace_utils.valeq(out1.trace, q_eta_z, strict=True)

    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    debug.seed(1)
    prp1 = Resample(Propose(
            proposal=Forward(enc_apg_z, enc_rws_eta),
            target=Reverse(og2, og2k)))

    fwd2 = Forward(kernel=enc_apg_eta, program=prp1)
    prp2 = Propose(proposal=fwd2, target=Reverse(og2, og2k))

    out2 = prp2(**kwargs)

    print(torch.equal(log_q_f, out2.proposal.log_joint))
    print(torch.equal(log_p_f, out2.target.log_joint))
    print(torch.equal(log_w_f, out2.log_weight))

    print(tensor_utils.show(log_w_f), tensor_utils.show(out2.log_weight))
    breakpoint();

    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f

    breakpoint();

    debug.seed(2)
    # ================================================================
    # sweep 2 (for m in range(num_sweeps-1)):
    # log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z, None, x=x, prior_ng=generative.prior_ng) ## forward kernel
    q_eta_z_f = q_eta_z_f.trace
    p_f = og.x_forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_eta(q_eta_z, None, x=x, prior_ng=generative.prior_ng)
    q_eta_z_b = q_eta_z_b.trace
    p_b = og.x_forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    breakpoint();
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_eta)
    log_w_z, q_eta_z, metrics = apg_update_z(enc_apg_z, og, q_eta_z, x)
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_z)

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

def train(objective, models, target, og, og2, data, assignments, num_epochs, sample_size, batch_size, normal_gamma_priors, with_tensorboard=False, lr=1e-3, seed=1, eval_break=50, is_smoketest=False, num_sweeps=None):
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
                loss, metrics = objective(enc_rws_eta=enc_rws_eta, enc_apg_z=enc_apg_z, generative=og2, og=og, x=x, sample_size=sample_size, num_sweeps=num_sweeps)

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
    # parser.add_argument('--num_sweeps', default=1, type=int)
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
    num_sweeps=2
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
    num_batches = (data.shape[0] // batch_size)
    sample_size = budget // num_sweeps

    # Models
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden=30)
    enc_apg_eta = Enc_apg_eta(K, D)
    generative = Generative(K, normal_gamma_priors)
    og = GenerativeOriginal(K, D, False, 'cpu')

    og2 = GenerativeV2(K, D, False, 'cpu')
    og2k = GenerativeLikelihoodV2()

    train(
        objective=rws_objective_eager if num_sweeps == 1 else apg_objective,
        models=[enc_rws_eta, enc_apg_z, enc_apg_eta],
        target=generative,
        og=og,
        og2=(og2, og2k),
        data=data,
        assignments=assignments,
        normal_gamma_priors=normal_gamma_priors,
        num_sweeps=num_sweeps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        sample_size=sample_size,
        is_smoketest=is_smoketest,
    )
    print(";".join([f'{k}={v}' for k, v in dict(objective=apg_objective, num_epochs=num_epochs, batch_size=batch_size, sample_size=sample_size, is_smoketest=is_smoketest).items()]))
    print('tada!')
