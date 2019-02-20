#!/usr/bin/env python3

import torch
from torch.distributions import Categorical, Dirichlet
from torch.nn.functional import softplus

import combinators
import gmm
import utils

class InitHmm(combinators.Primitive):
    def _forward(self, mu, sigma, pi0, **kwargs):
        pi = torch.zeros(*self.batch_shape, pi0.shape[-1], pi0.shape[-1])
        for k in range(pi0.shape[-1]):
            pi[:, k] = self.param_sample(Dirichlet, name='Pi_%d' % (k+1))
        z0 = self.sample(Categorical, softplus(pi[:, 0]), name='Z_0')
        return z0, mu, sigma, pi

class HmmStep(combinators.Primitive):
    def __init__(self, *args, **kwargs):
        super(HmmStep, self).__init__(*args, **kwargs)
        self.gmm = gmm.Gmm(batch_shape=self.batch_shape)

    def _forward(self, theta, t, data={}):
        z_prev, mu, sigma, pi = theta
        t += 1
        pi_prev = utils.particle_index(pi, z_prev)

        (z_current, _), p, _ = self.gmm(mu, sigma, pi_prev,
                                        latent_name='Z_%d' % t,
                                        observable_name='X_%d' % t, data=data)
        self.p = utils.join_traces(self.p, p['Gmm'])

        return z_current, mu, sigma, pi
