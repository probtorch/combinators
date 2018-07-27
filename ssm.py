#!/usr/bin/env python3

import probtorch
import torch
from torch.nn.functional import softplus

import combinators
import utils

def init_ssm(T=1, this=None):
    num_particles = this.trace.num_particles if hasattr(this.trace, 'num_particles')\
                    else 1

    params = this.args_vardict()
    mu = this.trace.param_normal(params, name='mu')
    sigma = torch.sqrt(this.trace.param_normal(params, name='sigma')**2)
    delta = this.trace.param_normal(params, name='delta')
    zs = torch.ones(num_particles, T+1, device=mu.device) * -1
    zs[:, 0] = this.trace.normal(mu, softplus(sigma), name='Z_0')
    return zs[:, 0], mu, sigma, delta

def ssm_step(theta, t, this=None):
    z_prev, mu, sigma, delta = theta
    t += 1
    z_current = this.trace.normal(z_prev + delta, softplus(sigma),
                                  name='Z_%d' % t)
    this.trace.normal(
        z_current, torch.ones(*z_current.shape, device=z_current.device),
        name='X_%d' % t,
        value=utils.optional_to(this.guide['X_%d' % t], z_current)
    )
    return z_current, mu, sigma, delta

def ssm_retrace(z_current, mu, sigma, delta, this=None):
    t = 1
    for key in this.trace:
        if 'Z_' in key:
            t = int(key[2:])
    z_current = this.trace['Z_%d' % t].value
    delta = this.trace['delta'].value
    mu = this.trace['mu'].value
    sigma = this.trace['sigma'].value
    return z_current, mu, sigma, delta
