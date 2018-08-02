#!/usr/bin/env python3

import probtorch
import torch
from torch.nn.functional import softplus

import combinators
import utils

def init_ssm(this=None):
    params = this.args_vardict()
    mu = this.trace.param_normal(params, name='mu')
    sigma = torch.sqrt(this.trace.param_normal(params, name='sigma')**2)
    delta = this.trace.param_normal(params, name='delta')
    z0 = this.trace.normal(mu, softplus(sigma), name='Z_0')
    return z0, mu, sigma, delta

def ssm_step(theta, t, this=None):
    z_prev, mu, sigma, delta = theta
    t += 1
    z_current = this.trace.normal(z_prev + delta, softplus(sigma),
                                  name='Z_%d' % t)
    this.trace.normal(
        z_current, torch.ones(*z_current.shape, device=z_current.device),
        name='X_%d' % t,
    )
    return z_current, mu, sigma, delta
