#!/usr/bin/env python3

import torch
from torch.distributions import Normal
from torch.nn.functional import softplus

def init_ssm(trace=None, params=None, data={}):
    mu = trace.param_sample(Normal, params, name='mu')
    sigma = torch.sqrt(trace.param_sample(Normal, params, name='sigma')**2)
    delta = trace.param_sample(Normal, params, name='delta')
    z0 = trace.sample(Normal, mu, softplus(sigma), name='Z_0')
    return z0, mu, sigma, delta

def ssm_step(theta, t, trace=None, data={}):
    z_prev, mu, sigma, delta = theta
    t += 1
    z_current = trace.sample(Normal, z_prev + delta, softplus(sigma),
                             name='Z_%d' % t)
    trace.variable(
        Normal, z_current,
        torch.ones(*z_current.shape, device=z_current.device), name='X_%d' % t,
        value=data.get('X_%d' % t)
    )
    return z_current, mu, sigma, delta
