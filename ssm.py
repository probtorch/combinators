#!/usr/bin/env python3

import torch
from torch.distributions import Normal
from torch.nn.functional import softplus

import combinators

class InitSsm(combinators.Primitive):
    @property
    def name(self):
        return 'InitSsm'

    def _forward(self, *args, **kwargs):
        mu = self.param_sample(Normal, name='mu')
        sigma = self.param_sample(Normal, name='sigma')
        delta = self.param_sample(Normal, name='delta')
        z0 = self.sample(Normal, mu, softplus(sigma), name='Z_0')
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
