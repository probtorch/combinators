#!/usr/bin/env python3

import torch
from torch.distributions import Dirichlet

import gmm
import utils

def init_hmm(trace=None, params=None):
    mu, sigma, pi0 = gmm.init_gmm('Pi_0', trace=trace, params=params)

    pi = torch.zeros(pi0.shape[0], pi0.shape[1], pi0.shape[1])
    for k in range(pi0.shape[1]):
        pi[:, k] = trace.param_sample(Dirichlet, params, name='Pi_%d' % (k+1))

    z0, _ = gmm.gmm(mu, sigma, pi0, latent_name='Z_0', observable_name=None,
                    trace=trace)
    return z0, mu, sigma, pi, pi0

def hmm_step(theta, t, trace=None):
    z_prev, mu, sigma, pi, pi0 = theta
    t += 1
    pi_prev = utils.particle_index(pi, z_prev)

    z_current, _ = gmm.gmm(mu, sigma, pi_prev, latent_name='Z_%d' % t,
                           observable_name='X_%d' % t, trace=trace)
    return z_current, mu, sigma, pi, pi0
