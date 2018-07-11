#!/usr/bin/env python3

import probtorch
from probtorch.util import log_sum_exp
import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus

import combinators
import utils

def init_hmm(T=1, this=None):
    num_particles = this.trace.num_particles\
                    if hasattr(this.trace, 'num_particles') else 1
    params = this.args_vardict()
    pi0 = this.trace.param_dirichlet(params, name='Pi_0')
    pi = torch.zeros(num_particles, pi0.shape[1], pi0.shape[1])
    for k in range(pi0.shape[1]):
        pi[:, k] = this.trace.param_dirichlet(params, name='Pi_%d' % (k+1))
    mu = this.trace.param_normal(params, name='mu')
    sigma = torch.sqrt(this.trace.param_normal(params, name='sigma')**2)
    z0 = this.trace.variable(Categorical, pi0, name='Z_0')
    this.trace.annotation('init_hmm', 'Z_0')['log_marginals'] = torch.log(pi0)
    return z0, mu, sigma, pi

def hmm_step(z_prev, mu, sigma, pi, t, this=None):
    t += 1
    z_current = this.trace.variable(Categorical,
                                    softplus(utils.particle_index(pi, z_prev)),
                                    name='Z_%d' % t)
    this.trace.normal(
        utils.particle_index(mu, z_current),
        softplus(utils.particle_index(sigma, z_current)),
        name='X_%d' % t,
        value=utils.optional_to(this.observations['X_%d' % t], mu)
    )
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
