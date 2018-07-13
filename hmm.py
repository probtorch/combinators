#!/usr/bin/env python3

import probtorch
from probtorch.util import log_sum_exp
import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus

import filtering
import gmm
import utils

def init_hmm(T=1, this=None):
    num_particles = this.trace.num_particles\
                    if hasattr(this.trace, 'num_particles') else None
    params = this.args_vardict()
    mu, sigma, pi0 = gmm.init_gmm('Pi_0', this)

    if num_particles:
        pi = torch.zeros(num_particles, pi0.shape[1], pi0.shape[1])
        for k in range(pi0.shape[1]):
            pi[:, k] = this.trace.param_dirichlet(params, name='Pi_%d' % (k+1))
    else:
        pi = torch.zeros(pi0.shape[0], pi0.shape[0])
        for k in range(pi0.shape[0]):
            pi[k] = this.trace.param_dirichlet(params, name='Pi_%d' % (k+1))

    z0, _ = gmm.gmm(mu, sigma, pi0, latent_name='Z_0', observable_name=None,
                    this=this)
    return z0, mu, sigma, pi, pi0

def forward_filter_hmm(mu, sigma, pi, pi0):
    transition = lambda prev, current: torch.log(pi[:, prev, current])
    observation_dists = [Normal(mu[:, state], softplus(sigma[:, state]))
                         for state in range(pi.shape[1])]
    return filtering.ForwardMessenger(hmm_step, 'Z_%d', 'X_%d', transition,
                                      observation_dists,
                                      initial_marginals=('init_hmm',
                                                         torch.log(pi0)))

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

    pis = [this.trace['Pi_%d' % (k+1)].value for k in range(mu.shape[1])]
    pi = torch.stack(pis, dim=1)

    return z_current, mu, sigma, pi
