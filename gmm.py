#!/usr/bin/env python3

import probtorch
from probtorch.util import log_sum_exp
import torch
from torch.distributions import Categorical, Dirichlet, Normal
from torch.nn.functional import softplus

import combinators
import utils

def init_gmm(pi_name='Pi', trace=None, params=None):
    pi = trace.param_sample(Dirichlet, params, name=pi_name)
    mu = trace.param_sample(Normal, params, name='mu')
    sigma = torch.sqrt(trace.param_sample(Normal, params, name='sigma')**2)
    return mu, sigma, pi

def gmm(mu, sigma, pi, latent_name='Z', observable_name='X', trace=None):
    z = trace.sample(Categorical, softplus(pi), name=latent_name)
    if observable_name:
        x = trace.sample(Normal, utils.particle_index(mu, z),
                         softplus(utils.particle_index(sigma, z)),
                         name=observable_name)
    else:
        x = None
    return z, x
