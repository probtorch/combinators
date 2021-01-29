# ============================================================================= #
#                              APG Objectives                                   #
# ============================================================================= #
import torch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch._six import inf


def kls_eta(models, ob, z):
    """
    compute the KL divergence KL(p(\eta | x, z)|| q(\eta | x, z))
    """
    (_, _, enc_apg_eta, generative) = models
    q_f_eta = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True)
    mu = q_f_eta['means'].value
    tau = q_f_eta['precisions'].value
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_f_eta['precisions'].dist.concentration
    q_beta = q_f_eta['precisions'].dist.rate
    q_mu = q_f_eta['means'].dist.loc
    q_std = q_f_eta['means'].dist.scale
    q_nu = 1. / (tau * (q_std**2)) # nu*tau = 1 / std**2

    posterior_alpha, posterior_beta, posterior_mu, posterior_nu = posterior_eta(ob=ob,
                                                                                z=z,
                                                                                prior_alpha=generative.prior_alpha,
                                                                                prior_beta=generative.prior_beta,
                                                                                prior_mu=generative.prior_mu,
                                                                                prior_nu=generative.prior_nu)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha=q_alpha,
                                   q_beta=q_beta,
                                   q_mu=q_mu,
                                   q_nu=q_nu,
                                   p_alpha=posterior_alpha,
                                   p_beta=posterior_beta,
                                   p_mu=posterior_mu,
                                   p_nu=posterior_nu)

    inckl = kl_eta_in.sum(-1).mean().detach()
    exckl = kl_eta_ex.sum(-1).mean().detach()
    return exckl, inckl

# def kl_eta_and_z(enc_apg_eta, enc_apg_z, generative, ob, z):
#     q_f_eta = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True)
#     mu = q_f_eta['means'].value
#     tau = q_f_eta['precisions'].value
#     ## KLs for mu and sigma based on Normal-Gamma prior
#     q_alpha = q_f_eta['precisions'].dist.concentration
#     q_beta = q_f_eta['precisions'].dist.rate
#     q_mu = q_f_eta['means'].dist.loc
#     q_std = q_f_eta['means'].dist.scale
#     q_nu = 1. / (tau * (q_std**2)) # nu*tau = 1 / std**2

#     q_f_z = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=True)
#     q_pi = q_f_z['states'].dist.probs

#     posterior_alpha, posterior_beta, posterior_mu, posterior_nu = posterior_eta(ob=ob,
#                                                                                 z=z,
#                                                                                 prior_alpha=generative.prior_alpha,
#                                                                                 prior_beta=generative.prior_beta,
#                                                                                 prior_mu=generative.prior_mu,
#                                                                                 prior_nu=generative.prior_nu)
#     kl_eta_ex, kl_eta_in = kls_NGs(q_alpha=q_alpha,
#                                    q_beta=q_beta,
#                                    q_mu=q_mu,
#                                    q_nu=q_nu,
#                                    p_alpha=posterior_alpha,
#                                    p_beta=posterior_beta,
#                                    p_mu=posterior_mu,
#                                    p_nu=posterior_nu)
#     posterior_logits = posterior_z(ob=ob,
#                                    tau=tau,
#                                    mu=mu,
#                                    prior_pi=generative.prior_pi)
#     kl_z_ex, kl_z_in = kls_cats(q_logits=q_pi.log(),
#                                 p_logits=posterior_logits)
#     inckls = {"inckl_eta" : kl_eta_in.sum(-1).mean(0).detach(),"inckl_z" : kl_z_in.sum(-1).mean(0).detach() }
#     return inckls

def params_to_nats(alpha, beta, mu, nu):
    """
    distribution parameters to natural parameters
    """
    return alpha - (1./2), - beta - (nu * (mu**2) / 2), nu * mu, - nu / 2

def nats_to_params(nat1, nat2, nat3, nat4):
    """
    natural parameters to distribution parameters
    """
    alpha = nat1 + (1./2)
    nu = -2 * nat4
    mu = nat3 / nu
    beta = - nat2 - (nu * (mu**2) / 2)
    return alpha, beta, mu, nu

def pointwise_sufficient_statistics(ob:Tensor, z:Tensor, expand1=True):
    """
    pointwise sufficient statstics
    stat1 : sum of I[z_n=k], S * B * K * 1
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    TODO: add some asserts for the comment above
    """
    stat1 = z.sum(2).unsqueeze(-1)
    if expand1:
        stat1_expand = stat1.repeat(1, 1, 1, ob.shape[-1]) ## S * B * K * D
        stat1_nonzero = stat1_expand
        stat1_nonzero[stat1_nonzero == 0.0] = 1.0
        stat1 = (stat1_expand, stat1_nonzero)

    z_expand = z.unsqueeze(-1).repeat(1, 1, 1, 1, ob.shape[-1])
    ob_expand = ob.unsqueeze(-1).repeat(1, 1, 1, 1, z.shape[-1]).transpose(-1, -2)
    stat2 = (z_expand * ob_expand).sum(2)
    stat3 = (z_expand * (ob_expand**2)).sum(2)
    return stat1, stat2, stat3

def posterior_eta(ob, z, prior_alpha, prior_beta, prior_mu, prior_nu):
    """
    conjugate postrior of eta, given the normal-gamma prior
    """
    (stat1_expand, stat1_nonzero), stat2, stat3 = pointwise_sufficient_statistics(ob, z)
    x_bar = stat2 / stat1_nonzero
    post_alpha = prior_alpha + stat1_expand / 2
    post_nu = prior_nu + stat1_expand
    post_mu = (prior_mu * prior_nu + stat2) / (stat1_expand + prior_nu)
    post_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1_expand * prior_nu / (stat1_expand + prior_nu)) * ((x_bar - prior_nu)**2) / 2.
    return post_alpha, post_beta, post_mu, post_nu

def posterior_z(ob, tau, mu, prior_pi):
    """
    posterior of z, given the Gaussian likelihood and the uniform prior
    """
    N = ob.shape[-2]
    K = mu.shape[-2]
    sigma = 1. / tau.sqrt()
    mu_expand = mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    sigma_expand = sigma.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    ob_expand = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    log_gammas = Normal(mu_expand, sigma_expand).log_prob(ob_expand).sum(-1).transpose(-1, -2) + prior_pi.log() # S * B * N * K
    post_logits = F.softmax(log_gammas, dim=-1).log()
    return post_logits

## some standard KL-divergence functions
def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def kls_normals(q_mean, q_sigma, p_mean, p_sigma):
    Kl_ex = kl_normal_normal(q_mean, q_sigma, p_mean, p_sigma).sum(-1)
    Kl_in = kl_normal_normal(p_mean, p_sigma, q_mean, q_sigma).sum(-1)
    return Kl_ex, Kl_in

def kl_gamma_gamma(p_alpha, p_beta, q_alpha, q_beta):
    t1 = q_alpha * (p_beta / q_beta).log()
    t2 = torch.lgamma(q_alpha) - torch.lgamma(p_alpha)
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha)
    t4 = (q_beta - p_beta) * (p_alpha / p_beta)
    return t1 + t2 + t3 + t4

def kls_gammas(q_alpha, q_beta, p_alpha, p_beta):
    KL_ex = kl_gamma_gamma(q_alpha, q_beta, p_alpha, p_beta).sum(-1)
    KL_in = kl_gamma_gamma(p_alpha, p_beta, q_alpha, q_beta).sum(-1)
    return KL_ex, KL_in


def kl_NG_NG(p_alpha, p_beta, p_mu, p_nu, q_alpha, q_beta, q_mu, q_nu):
    diff = q_mu - p_mu
    t1 = (1. / 2) * ((p_alpha / p_beta) *  (diff ** 2) * q_nu + (q_nu / p_nu) - (torch.log(q_nu) - torch.log(p_nu)) - 1)
    t2 = q_alpha * (torch.log(p_beta) - torch.log(q_beta)) - (torch.lgamma(p_alpha) - torch.lgamma(q_alpha))
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha) - (p_beta - q_beta) * p_alpha / p_beta
    return t1 + t2 + t3

def kls_NGs(q_alpha, q_beta, q_mu, q_nu, p_alpha, p_beta, p_mu, p_nu):
    kl_ex = kl_NG_NG(q_alpha, q_beta, q_mu, q_nu, p_alpha, p_beta, p_mu, p_nu).sum(-1)
    kl_in = kl_NG_NG(p_alpha, p_beta, p_mu, p_nu, q_alpha, q_beta, q_mu, q_nu).sum(-1)
    return kl_ex, kl_in


def kl_cat_cat(p_logits, q_logits, EPS=-1e14):
    p_probs= p_logits.exp()
    ## To prevent from infinite KL due to ill-defined support of q
    q_logits[q_logits == -inf] = EPS
    t = p_probs * (p_logits - q_logits)
    # t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(q_logits, p_logits):
    KL_ex = kl_cat_cat(q_logits, p_logits)
    KL_in = kl_cat_cat(p_logits, q_logits)
    return KL_ex, KL_in

def apg_objective(models, x, result_flags, num_sweeps, block, resampler):
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
    trace = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_apg_z, generative, x, trace, result_flags)
    tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
    for m in range(num_sweeps-1):
        if block == 'decomposed':
            log_w_eta, tau, mu, trace = apg_update_eta(enc_apg_eta, generative, x, z, tau, mu, trace, result_flags)
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_eta)
            log_w_z, z, trace = apg_update_z(enc_apg_z, generative, x, tau, mu, z, trace, result_flags)
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_z)
        elif block == 'joint':
            log_w, tau, mu, z, trace = apg_update_joint(enc_apg_z, enc_apg_eta, generative, x, z, tau, mu, trace, result_flags)
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
        else:
            raise ValueError
    if result_flags['loss_required']:
        trace['loss'] = torch.cat(trace['loss'], 0)
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0)
    if result_flags['mode_required']:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0)
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)
        trace['E_z'] = torch.cat(trace['E_z'], 0)  # (num_sweeps) * B * N * K
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def rws_objective(models, x, result_flags):
    """
    The objective of RWS method
    """
    trace = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []}
    (enc_rws_eta, enc_rws_z, generative) = models
    w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    if result_flags['loss_required']:
        trace['loss'] = torch.cat(trace['loss'], 0)
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0)
    if result_flags['mode_required']:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0)
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)
        trace['E_z'] = torch.cat(trace['E_z'], 0)
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0)
    return trace

def oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags):
    """
    One-shot for eta and z, like a normal RWS
    """
    q_eta = enc_rws_eta(x, prior_ng=generative.prior_ng, sampled=True)
    p_eta = generative.eta_prior(q=q_eta)
    log_q_eta = q_eta['means'].log_prob.sum(-1).sum(-1) + q_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_eta = p_eta['means'].log_prob.sum(-1).sum(-1) + p_eta['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_eta['precisions'].value
    mu = q_eta['means'].value
    q_z = enc_rws_z(x, tau=tau, mu=mu, sampled=True)
    p_z = generative.z_prior(q=q_z)
    log_q_z = q_z['states'].log_prob.sum(-1)
    log_p_z = p_z['states'].log_prob.sum(-1)
    z = q_z['states'].value
    ll = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_p = ll + log_p_eta + log_p_z
    log_q =  log_q_eta + log_q_z
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta['means'].dist.loc.mean(0).detach()
        E_z = q_z['states'].dist.probs.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        log_joint = log_p.detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return log_w, tau, mu, z, trace

def apg_update_joint(enc_apg_z, enc_apg_eta, generative, x, z_old, tau_old, mu_old, trace, result_flags):
    """
    Jointly update all the variables
    """
    q_f_eta = enc_apg_eta(x, z=z_old, prior_ng=generative.prior_ng, sampled=True)
    p_f_eta = generative.eta_prior(q=q_f_eta)
    log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f_eta = p_f_eta['means'].log_prob.sum(-1).sum(-1) + p_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f_eta['precisions'].value
    mu = q_f_eta['means'].value
    q_f_z = enc_apg_z(x, tau=tau, mu=mu, sampled=True)
    p_f_z = generative.z_prior(q=q_f_z)
    log_q_f_z = q_f_z['states'].log_prob.sum(-1)
    log_p_f_z = p_f_z['states'].log_prob.sum(-1)
    z = q_f_z['states'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f_eta - log_q_f_eta - log_q_f_z + log_p_f_z
    ## backward
    q_b_z = enc_apg_z(x, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b_z = generative.z_prior(q=q_b_z)
    log_q_b_z = q_b_z['states'].log_prob.sum(-1)
    log_p_b_z = p_b_z['states'].log_prob.sum(-1)
    q_b_eta = enc_apg_eta(x, z=z_old, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b_eta = generative.eta_prior(q=q_b_eta)
    log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).sum(-1) + q_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b_eta = p_b_eta['means'].log_prob.sum(-1).sum(-1) + p_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(x, z=z_old, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b_eta - log_q_b_eta + log_p_b_z - log_q_b_z
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f_eta - log_q_f_z)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(z.unsqueeze(0))
    if result_flags['density_required']:
        log_joint = (ll_f + log_p_f_eta + log_p_f_z).detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return log_w, tau, mu, z, trace


def apg_update_eta(enc_apg_eta, generative, x, z, tau_old, mu_old, trace, result_flags):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = enc_apg_eta(x, z=z, prior_ng=generative.prior_ng, sampled=True) ## forward kernel
    p_f = generative.eta_prior(q=q_f)
    log_q_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f = p_f['means'].log_prob.sum(-1).sum(-1) + p_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_eta(x, z=z, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b = generative.eta_prior(q=q_b)
    log_q_b = q_b['means'].log_prob.sum(-1).sum(-1) + q_b['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b = p_b['means'].log_prob.sum(-1).sum(-1) + p_b['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(x, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if result_flags['mode_required']:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace

def apg_update_z(enc_apg_z, generative, x, tau, mu, z_old, trace, result_flags):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable state i.e. z
    """
    q_f = enc_apg_z(x, tau=tau, mu=mu, sampled=True)
    p_f = generative.z_prior(q=q_f)
    log_q_f = q_f['states'].log_prob
    log_p_f = p_f['states'].log_prob
    z = q_f['states'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=False)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_z(x, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b = generative.z_prior(q=q_b)
    log_q_b = q_b['states'].log_prob
    log_p_b = p_b['states'].log_prob
    ll_b = generative.log_prob(x, z=z_old, tau=tau, mu=mu, aggregate=False)
    log_w_b = ll_b + log_p_b - log_q_b
    log_w_local = (log_w_f - log_w_b).detach()
    w_local = F.softmax(log_w_local, 0).detach()
    log_w = log_w_local.sum(-1)
    if result_flags['loss_required']:
        loss = (w_local * (- log_q_f)).sum(0).sum(-1).mean()
        trace['loss'][-1] = trace['loss'][-1] + loss.unsqueeze(0)
    if result_flags['mode_required']:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'][-1] = trace['density'][-1] + (ll_f + log_p_f).sum(-1).unsqueeze(0)
    return log_w, z, trace

def resample_variables(resampler, tau, mu, z, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    return tau, mu, z


def gibbs_objective(models, x, result_flags, num_sweeps):
    """
    The Gibbs sampler objective
    """
    trace = {'density' : []}
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    for m in range(num_sweeps-1):
        tau, mu, z, trace = gibbs_sweep(generative, x, z, trace)
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def gibbs_sweep(generative, x, z, trace):
    """
    Gibbs updates
    """
    post_alpha, post_beta, post_mu, post_nu = posterior_eta(x,
                                                            z=z,
                                                            prior_alpha=generative.prior_alpha,
                                                            prior_beta=generative.prior_beta,
                                                            prior_mu=generative.prior_mu,
                                                            prior_nu=generative.prior_nu)

    E_tau = (post_alpha / post_beta).mean(0)
    E_mu = post_mu.mean(0)
    tau = Gamma(post_alpha, post_beta).sample()
    mu = Normal(post_mu, 1. / (post_nu * tau).sqrt()).sample()
    posterior_logits = posterior_z(x, tau, mu, generative.prior_pi)
    E_z = posterior_logits.exp().mean(0)
    z = cat(logits=posterior_logits).sample()
    ll = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_prior_tau = Gamma(generative.prior_alpha, generative.prior_beta).log_prob(tau).sum(-1).sum(-1)
    log_prior_mu = Normal(generative.prior_mu, 1. / (generative.prior_nu * tau).sqrt()).log_prob(mu).sum(-1).sum(-1)
    log_prior_z = cat(probs=generative.prior_pi).log_prob(z).sum(-1)
    log_joint = ll + log_prior_tau + log_prior_mu + log_prior_z
    trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return tau, mu, z, trace

def hmc_objective(models, x, result_flags, hmc_sampler):
    """
    HMC + marginalization over discrete variables in GMM problem
    """
    trace = {'density' : []}
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    log_tau, mu, trace = hmc_sampler.hmc_sampling(generative,
                                                  x,
                                                  log_tau=tau.log(),
                                                  mu=mu,
                                                  trace=trace)
    trace['density'] = torch.cat(trace['density'], 0)
    return log_tau.exp(), mu, trace

def bpg_objective(models, x, result_flags, num_sweeps, resampler):
    """
    bpg objective
    """
    trace = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_apg_z, generative, x, trace, result_flags)
    tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
    for m in range(num_sweeps-1):
        log_w_eta, tau, mu, trace = bpg_update_eta(generative, x, z, tau, mu, trace)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_eta)
        log_w_z, z, trace = apg_update_z(enc_apg_z, generative, x, tau, mu, z, trace, result_flags)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_z)
    trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def bpg_update_eta(generative, x, z, tau_old, mu_old, trace):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = generative.eta_sample_prior(S=x.shape[0], B=x.shape[1])
    ## Not typo, here p is q since we sample from prior
    log_p_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    ll_b = generative.log_prob(x, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w = (ll_f - ll_b).detach()
    trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace
