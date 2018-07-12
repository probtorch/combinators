#!/usr/bin/env python3

import probtorch
from probtorch.util import log_sum_exp
import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus

import combinators
import gmm
import utils

def init_hmm(T=1, this=None):
    num_particles = this.trace.num_particles\
                    if hasattr(this.trace, 'num_particles') else 1
    params = this.args_vardict()
    mu, sigma, pi0 = gmm.init_gmm('Pi_0', this)

    pi = torch.zeros(num_particles, pi0.shape[1], pi0.shape[1])
    for k in range(pi0.shape[1]):
        pi[:, k] = this.trace.param_dirichlet(params, name='Pi_%d' % (k+1))

    z0, _ = gmm.gmm(mu, sigma, pi0, latent_name='Z_0', observable_name=None,
                    this=this)
    this.trace.annotation('init_hmm', 'Z_0')['log_marginals'] = torch.log(pi0)
    return z0, mu, sigma, pi

def hmm_step(z_prev, mu, sigma, pi, t, this=None):
    t += 1
    pi_prev = utils.particle_index(pi, z_prev)

    z_current, _ = gmm.gmm(mu, sigma, pi_prev, latent_name='Z_%d' % t,
                           observable_name='X_%d' % t, this=this)
    return z_current, mu, sigma, pi

def hmm_retrace(z_current, mu, sigma, pi, this=None):
    t = 1
    for key in reversed(list(this.trace)):
        if 'Z_' in key:
            t = int(key[2:])
            break

    z_current = this.trace['Z_%d' % t].value
    mu = this.trace['mu'].value
    sigma = this.trace['sigma'].value

    num_states = mu.shape[1]

    pis = [this.trace['Pi_%d' % (k+1)].value for k in range(mu.shape[1])]
    pi = torch.stack(pis, dim=1)

    prev_note = this.trace.annotation('hmm_step', 'Z_%d' % (t-1)) if t > 1 else\
                this.trace.annotation('init_hmm', 'Z_0')
    current_note = this.trace.annotation('hmm_step', 'Z_%d' % t)
    marginals = torch.zeros(pi.shape[0], num_states, num_states)
    for prev in range(num_states):
        for current in range(num_states):
            marginals[:, current, prev] = torch.log(pi[:, prev, current]) +\
                                          prev_note['log_marginals'][:, prev]
    marginals = log_sum_exp(marginals, dim=-1)
    current_note['log_marginals'] = marginals

    observation_note = this.trace.annotation('hmm_step', 'X_%d' % t)
    obs_marginal = torch.zeros(pi.shape[0], num_states)
    for current in range(num_states):
        counterfactual = Normal(mu[:, current], softplus(sigma[:, current]))
        observation = this.trace['X_%d' % t].value
        obs_marginal[:, current] = counterfactual.log_prob(observation) +\
                                   marginals[:, current]
    observation_note['log_marginals'] = log_sum_exp(obs_marginal, dim=-1)

    return z_current, mu, sigma, pi
