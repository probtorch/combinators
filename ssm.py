#!/usr/bin/env python3

import probtorch
import torch
from torch.nn.functional import softplus

import combinators
import utils

def init_ssm(T=1, trace=probtorch.Trace(), params={}):
    num_particles = trace.num_particles if hasattr(trace, 'num_particles')\
                    else 1

    mu = trace.param_normal(params, name='mu')
    sigma = torch.sqrt(trace.param_normal(params, name='sigma')**2)
    delta = trace.param_normal(params, name='delta')
    zs = torch.ones(num_particles, T+1, device=mu.device) * -1
    zs[:, 0] = trace.normal(mu, softplus(sigma), name='Z_0')
    return zs[:, 0], mu, sigma, delta

def ssm_step(z_prev, mu, sigma, delta, t, trace={},
             conditions=utils.EMPTY_TRACE):
    t += 1
    z_current = trace.normal(z_prev + delta, softplus(sigma), name='Z_%d' % t)
    trace.normal(z_current, torch.ones(*z_current.shape,
                                       device=z_current.device),
                 name='X_%d' % t,
                 value=utils.optional_to(conditions['X_%d' % t], z_current))
    return z_current, mu, sigma, delta, trace

def ssm_retrace(z_current, mu, sigma, delta, trace={},
                conditions=utils.EMPTY_TRACE):
    t = 1
    for key in trace:
        if 'Z_' in key:
            t = int(key[2:])
    z_current = trace['Z_%d' % t].value
    delta = trace['delta'].value
    mu = trace['mu'].value
    sigma = trace['sigma'].value
    return z_current, mu, sigma, delta, trace
