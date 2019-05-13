#!/usr/bin/env python3

import torch
from torch.distributions import Normal
from torch.nn.functional import softplus

from combinators.model import model

class InitSsm(model.Primitive):
    @property
    def name(self):
        return 'InitSsm'

    def _forward(self, *args, **kwargs):
        mu = self.param_sample(Normal, name='mu')
        sigma = self.param_sample(Normal, name='sigma')
        delta = self.param_sample(Normal, name='delta')
        z0 = self.sample(Normal, mu, softplus(sigma), name='Z_0')
        return z0, mu, sigma, delta

class SsmStep(model.Primitive):
    @property
    def name(self):
        return 'SsmStep'

    def _forward(self, theta, t, data={}):
        z_prev, mu, sigma, delta = theta
        t += 1
        z_current = self.sample(Normal, z_prev + delta, softplus(sigma),
                                name='Z_%d' % t)
        self.observe('X_%d' % t, data.get('X_%d' % t), Normal, z_current,
                     torch.ones(*z_current.shape, device=z_current.device))
        return z_current, mu, sigma, delta
