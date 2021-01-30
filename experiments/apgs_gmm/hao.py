#!/usr/bin/env python3

from combinators.stochastic import Trace
import torch.nn.functional as F
import combinators.debug as debug
import torch


def oneshot(enc_rws_eta, enc_apg_z, generative, x, metrics):
    """
    One-shot for eta and z, like a normal RWS
    """
    q_eta_z = enc_rws_eta(x, prior_ng=generative.prior_ng)
    q_eta_z = enc_apg_z(q_eta_z.trace, q_eta_z.output, x=x, prior_ng=generative.prior_ng, ix='')
    log_q = q_eta_z.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)

    p = generative.x_forward(q_eta_z.trace, x)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) ## it is annoying to repeat these same arguments every time I call .log_joint
    log_q = q_eta_z.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()

    if True: # result_flags['loss_required']:
        loss = (w * (- log_q)).sum(0).mean()
        metrics['loss'].append(loss.unsqueeze(0))
    if True: # result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0))
    if False: # result_flags['mode_required']:
        E_tau = (q_eta_z['precisions'].dist.concentration / q_eta_z['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta_z['means'].dist.loc.mean(0).detach()
        E_z = q_eta_z['states'].dist.probs.mean(0).detach()
        metrics['E_tau'].append(E_tau.unsqueeze(0))
        metrics['E_mu'].append(E_mu.unsqueeze(0))
        metrics['E_z'].append(E_z.unsqueeze(0))
    if True: # result_flags['density_required']:
        log_joint = log_p.detach()
        metrics['density'].append(log_joint.unsqueeze(0))
    return loss, log_w, q_eta_z, metrics

def apg_update_eta(enc_apg_eta, generative, q_eta_z_trace, x, metrics, result_flags):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z_trace, cond_outs=None, x=x, prior_ng=generative.prior_ng, ix='') ## forward kernel
    p_f = generative.x_forward(q_eta_z_f.trace, x)
    log_q_f = q_eta_z_f.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_eta(q_eta_z_trace, cond_outs=None, x=x, prior_ng=generative.prior_ng, ix='')
    p_b = generative.x_forward(q_eta_z_b.trace, x)
    log_q_b = q_eta_z_b.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
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
    aux = dict(
        log_q_f = log_q_f,
        log_p_f = log_p_f,
        log_w_f = log_w_f,
        q_eta_z_f = q_eta_z_f.trace,
        p_f = p_f,

        log_q_b = log_q_b,
        log_p_b = log_p_b,
        log_w_b = log_w_b,
        q_eta_z_b = q_eta_z_b.trace,
        p_b = p_b,
    )
    return log_w, q_eta_z_f, metrics, aux

def apg_update_z(enc_apg_z, generative, q_eta_z_trace, x, metrics, result_flags):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable state i.e. z
    """
    assert isinstance(q_eta_z_trace, Trace)
    # forward
    q_eta_z_f = enc_apg_z(q_eta_z_trace, cond_outs=None, x=x, ix='')
    p_f = generative.x_forward(q_eta_z_f.trace, x)
    log_q_f = q_eta_z_f.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_z(q_eta_z_trace, cond_outs=None, x=x, ix='')
    p_b = generative.x_forward(q_eta_z_b.trace, x)
    log_q_b = q_eta_z_b.trace.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
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

    aux = dict(
        log_q_f = log_q_f,
        log_p_f = log_p_f,
        log_w_f = log_w_f,
        q_eta_z_f = q_eta_z_f.trace,
        p_f = p_f,

        log_q_b = log_q_b,
        log_p_b = log_p_b,
        log_w_b = log_w_b,
        q_eta_z_b = q_eta_z_b.trace,
        p_b = p_b,
    )
    return log_w, q_eta_z_f, metrics, aux


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

def apg_objective(models, x, num_sweeps, resampler, resample=False):
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
    ob : S * B * N * D, observations, as data points
    tau: S * B * K * D, cluster precisions, as global variables
    mu: S * B * K * D, cluster means, as global variables
    eta := {tau, mu} global block
    z : S * B * N * K, cluster assignments, as local variables
    ==========
    """
    debug.seed(0)
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    _, log_w, q_eta_z, metrics = oneshot(enc_rws_eta, enc_apg_z, generative, x, metrics)
    q_eta_z_trace = q_eta_z.trace
    if resample:
        q_eta_z_trace = resample_variables(resampler, q_eta_z, log_weights=log_w)

    sweeps = [ dict(log_w=log_w, q_eta_z=q_eta_z, metrics=metrics) ]

    for m in range(num_sweeps-1):
        sweeps.append([dict(), dict()])
        log_w_eta, q_eta_z, metrics, aux = apg_update_eta(enc_apg_eta, generative, q_eta_z_trace, x, metrics, result_flags)
        q_eta_z_trace = q_eta_z.trace
        sweeps[-1][0] = dict(log_w_eta=log_w_eta, q_eta_z=q_eta_z_trace, metrics=metrics, aux=aux)
        if resample:
            q_eta_z_trace = resample_variables(resampler, q_eta_z_trace, log_weights=log_w_eta)
        log_w_z, q_eta_z, metrics, aux = apg_update_z(enc_apg_z, generative, q_eta_z_trace, x, metrics, result_flags)
        q_eta_z_trace = q_eta_z.trace
        sweeps[-1][1] = dict(log_w_z=log_w_z, q_eta_z=q_eta_z, metrics=metrics, aux=aux)
        if resample:
            q_eta_z_trace = resample_variables(resampler, q_eta_z_trace, log_weights=log_w_z)


    if result_flags['loss_required']:
        metrics['loss'] = torch.cat(metrics['loss'], 0)
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0)
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0)  # (num_sweeps) * S * B

    return sweeps, metrics
