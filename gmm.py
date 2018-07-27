#!/usr/bin/env python3

import probtorch
from probtorch.util import log_sum_exp
import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus

import combinators
import utils

def init_gmm(pi_name='Pi', this=None):
    params = this.args_vardict()
    pi = this.trace.param_dirichlet(params, name=pi_name)
    mu = this.trace.param_normal(params, name='mu')
    sigma = torch.sqrt(this.trace.param_normal(params, name='sigma')**2)
    return mu, sigma, pi

def gmm(mu, sigma, pi, latent_name='Z', observable_name='X', this=None):
    z = this.trace.variable(Categorical, softplus(pi), name=latent_name)
    if observable_name:
        x = this.trace.normal(
            utils.particle_index(mu, z),
            softplus(utils.particle_index(sigma, z)), name=observable_name,
            value=utils.optional_to(this.guide[observable_name], mu)
        )
    else:
        x = None
    return z, x
