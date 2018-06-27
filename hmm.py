#!/usr/bin/env python3

import probtorch
import torch

import combinators
import utils

def init_hmm(num_states, T=1, trace=probtorch.Trace(), params={}):
    num_particles = trace.num_particles if hasattr(trace, 'num_particles')\
                    else 1

    pi0 = trace.param_dirichlet(params, name='\\Pi_0')
    pi = torch.zeros(num_particles, num_states, num_states)
    for k in range(num_states):
        pi[:, k] = trace.param_dirichlet(params, name='\\Pi_%d' % (k+1))
    mu = trace.param_normal(params, name='\\mu')
    sigma = torch.sqrt(trace.param_normal(params, name='\\sigma')**2)
    zs = torch.ones(num_particles, T+1, dtype=torch.long) * -1
    zs[:, 0] = trace.variable(torch.distributions.Categorical, pi0, name='Z_0')
    return zs, pi, mu, sigma

def hmm_step(zs, pi, mu, sigma, t, trace={}, conditions=utils.EMPTY_TRACE):
    zs[:, t] = trace.variable(torch.distributions.Categorical,
                              utils.particle_index(pi, zs[:, t-1]),
                              name='Z_%d' % t)
    trace.normal(utils.particle_index(mu, zs[:, t]),
                 utils.particle_index(sigma, zs[:, t]),
                 name='X_%d' % t, value=conditions['X_%d' % t])
    return zs, pi, mu, sigma, trace

def hmm_retrace(zs, pi, mu, sigma, trace={}, conditions=utils.EMPTY_TRACE):
    t = 1
    for key in trace:
        if 'Z_' in key:
            t = int(key[2:])
    for step in range(t):
        zs[:, step] = trace['Z_%d' % step].value
    for k in range(pi.shape[1]):
        pi[:, k] = trace['\\Pi_%d' % k].value
    return zs, pi, mu, sigma, trace
