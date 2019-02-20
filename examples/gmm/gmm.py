#!/usr/bin/env python3

from torch.distributions import Categorical, Dirichlet, Normal
from torch.nn.functional import softplus

import combinators.model as model
import combinators.utils as utils

class InitGmm(model.Primitive):
    def _forward(self, pi_name='Pi', **kwargs):
        pi = self.param_sample(Dirichlet, name=pi_name)
        mu = self.param_sample(Normal, name='mu')
        sigma = self.param_sample(Normal, name='sigma')
        return mu, sigma, pi

class Gmm(model.Primitive):
    def _forward(self, mu, sigma, pi, latent_name='Z', observable_name='X',
                 data={}):
        z = self.sample(Categorical, softplus(pi), name=latent_name)
        if observable_name:
            x = self.observe(observable_name, data.get(observable_name),
                             Normal, utils.particle_index(mu, z),
                             softplus(utils.particle_index(sigma, z)))
        else:
            x = None
        return z, x
