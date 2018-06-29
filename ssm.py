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
    zs = torch.ones(num_particles, T+1) * -1
    zs[:, 0] = trace.normal(mu, softplus(sigma), name='Z_0')
    return zs, mu, sigma, delta

def ssm_step(zs, mu, sigma, delta, t, trace={}, conditions=utils.EMPTY_TRACE):
    zs[:, t] = zs[:, t-1] + trace.normal(delta, softplus(sigma),
                                         name='Z_%d' % t)
    trace.normal(zs[:, t], torch.ones(*zs[:, t].shape), name='X_%d' % t,
                 value=conditions['X_%d' % t])
    return zs, mu, sigma, delta, trace

def ssm_retrace(zs, mu, sigma, delta, trace={}, conditions=utils.EMPTY_TRACE):
    t = 1
    for key in trace:
        if 'Z_' in key:
            t = int(key[2:])
    for step in range(t):
        zs[:, step] = trace['Z_%d' % step].value
    delta = trace['delta'].value
    return zs, mu, sigma, delta, trace
