#!/usr/bin/env python3

import probtorch
import torch
from torch.distributions import Categorical
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
    pis = [this.trace['Pi_%d' % (k+1)].value for k in range(mu.shape[1])]
    pi = torch.stack(pis, dim=1)
    return z_current, mu, sigma, pi
