#!/usr/bin/env python3

import torch
from torch.distributions import Categorical, Dirichlet, MultivariateNormal
from torch.distributions import Normal
from torch.distributions.transforms import LowerCholeskyTransform
import torch.nn as nn
from torch.nn.functional import softplus

import combinators
import utils

def reflect_directions(dir_loc):
    dir_locs = dir_loc.unsqueeze(-2).repeat(1, 4, 1)
    dir_locs[:, 1, 1] *= -1
    dir_locs[:, 2, :] *= -1
    dir_locs[:, 3, 0] *= -1
    return dir_locs / (dir_locs**2).sum(dim=-1).unsqueeze(-1).sqrt()

class InitBouncingBall(combinators.Primitive):
    def _forward(self, data={}):
        initial_alpha = self.param_sample(Dirichlet, name='alpha_0')
        pi = self.sample(Dirichlet, initial_alpha, name='Pi')
        initial_z = self.sample(Categorical, pi, name='direction_0')

        transition_alpha = torch.stack([
            self.param_sample(Dirichlet, name='alpha_%d' % (d+1))
            for d in range(4)
        ], dim=1)
        transition = torch.stack([
            self.sample(Dirichlet, transition_alpha[:, d], name='A_%d' % (d+1))
            for d in range(4)
        ], dim=1)

        dir_locs = self.param_sample(Normal, name='directions__loc')
        dir_locs = reflect_directions(dir_locs)
        dir_covs = self.param_sample(Normal, name='directions__cov')

        params = self.args_vardict()
        initial_position = data['position_0'].expand(
            params['position_0']['loc'].shape
        )
        initial_position = self.observe('position_0', initial_position,
                                        Normal, params['position_0']['loc'],
                                        softplus(params['position_0']['scale']))

        return initial_position, initial_z, transition, dir_locs, dir_covs

class BouncingBallStep(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        position, z_prev, transition, dir_locs, dir_covs = theta
        directions = {
            'loc': dir_locs,
            'covariance_matrix': dir_covs,
        }
        t += 1

        transition_prev = utils.particle_index(transition, z_prev)
        z_current = self.sample(Categorical, transition_prev,
                                name='direction_%d' % t)
        direction = utils.vardict_particle_index(directions, z_current)
        direction_covariance = direction['covariance_matrix']

        velocity = self.observe(
            'displacement_%d' % t, data.get('displacement_%d' % t),
            MultivariateNormal, direction['loc'],
            scale_tril=LowerCholeskyTransform()(direction_covariance),
        )
        position = position + velocity

        return position, z_current, transition, dir_locs, dir_covs

class ProposalStep(combinators.Primitive):
    def __init__(self, *args, name=None, **kwargs):
        super(ProposalStep, self).__init__(*args, **kwargs)
        self._name = name
        self.direction_predictor = nn.Sequential(
            nn.Linear(2, 4),
            nn.Softsign(),
            nn.Linear(4, 4),
            nn.LogSoftmax(dim=-1),
        )

    @property
    def name(self):
        if self._name:
            return self._name
        return super(ProposalStep, self).name

    def cond(self, qs):
        result = ProposalStep(name=self.name, params=self.args_vardict(False),
                              trainable=self._hyperparams_trainable,
                              batch_shape=self.batch_shape, q=qs[self.name])
        result.direction_predictor = self.direction_predictor
        return result

    def _forward(self, theta, t, data={}):
        position, _, transition, dir_locs, dir_covs = theta
        directions = {
            'loc': dir_locs,
            'covariance_matrix': dir_covs,
        }
        t += 1

        direction_predictions = self.direction_predictor(
            data.get('displacement_%d' % t)
        )
        direction_predictions = direction_predictions.expand(
            position.shape[0], 4
        )
        z_prev = Categorical(logits=direction_predictions).sample()
        transition_prev = utils.particle_index(transition, z_prev)
        z_current = self.sample(Categorical, transition_prev,
                                name='direction_%d' % t)

        direction = utils.vardict_particle_index(directions, z_current)
        direction_covariance = direction['covariance_matrix']
        velocity_name = 'displacement' if self.training else 'velocity'
        velocity_name += '_%d' % t
        if self.training:
            velocity = self.sample(
                MultivariateNormal, loc=direction['loc'],
                scale_tril=LowerCholeskyTransform()(direction_covariance),
                name='displacement_%d' % t,
            )
        else:
            velocity = self.sample(
                MultivariateNormal, loc=direction['loc'],
                scale_tril=LowerCholeskyTransform()(direction_covariance),
                name='velocity_%d' % t
            )
        position = position + velocity
        return position, z_current, transition, dir_locs, dir_covs
