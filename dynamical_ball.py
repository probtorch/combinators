#!/usr/bin/env python3

import torch
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators

class InitBallDynamics(combinators.Primitive):
    def __init__(self, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'boundary': {
                'loc': torch.zeros([]),
                'scale': torch.ones([]),
            },
            'dynamics': {
                'loc': torch.eye(2),
                'scale': torch.ones(2, 2),
            },
            'uncertainty': {
                'loc': torch.ones(2),
                'scale': torch.ones(2),
            },
            'noise': {
                'loc': torch.ones(2),
                'scale': torch.ones(2),
            },
            'position_0': {
                'loc': torch.ones(2),
                'covariance_matrix': torch.eye(2),
            },
        } if not params else params
        super(InitBallDynamics, self).__init__(params, trainable, batch_shape,
                                               q)
    def _forward(self, data={}):
        boundary = self.param_sample(LogNormal, name='boundary')
        dynamics = self.param_sample(Normal, name='dynamics')
        uncertainty = self.param_sample(LogNormal, name='uncertainty')
        noise = self.param_sample(LogNormal, name='noise')
        pos_params = self.args_vardict()['position_0']
        pos_scale = LowerCholeskyTransform()(pos_params['covariance_matrix'])
        position = self.sample(MultivariateNormal, loc=pos_params['loc'],
                               scale_tril=pos_scale, name='position_0')
        return boundary, dynamics, uncertainty, noise, position

class StepBallDynamics(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        boundary, dynamics, uncertainty, noise, position = theta
        t += 1

        position = position + self.sample(
            Normal, torch.bmm(dynamics, position.unsqueeze(-1)).squeeze(-1),
            softplus(uncertainty), name='velocity_%d' % t
        )

        overage = softplus(position[:, 0] - boundary)
        position[:, 0] = torch.where(overage > torch.zeros(overage.shape),
                                     boundary - overage, position[:, 0])
        dynamics[:, 0, :] = torch.where(overage > torch.zeros(overage.shape),
                                        -dynamics[: 0, :], dynamics[:, 0, :])
        overage = torch.norm(boundary[:, 2, 0] - position[:, 0], dim=-1)
        position[:, 0] = torch.where(overage > torch.zeros(overage.shape),
                                     boundary[:, 2, 0] + overage,
                                     position[:, 0])
        dynamics[:, 0, :] = torch.where(overage > torch.zeros(overage.shape),
                                        -dynamics[: 0, :], dynamics[:, 0, :])

        overage = torch.norm(position[:, 1] - boundary[:, 3, 1], dim=-1)
        position[:, 1] = torch.where(overage > torch.zeros(overage.shape),
                                     boundary[:, 3, 1] - overage,
                                     position[:, 1])
        dynamics[:, 1, :] = torch.where(overage > torch.zeros(overage.shape),
                                        -dynamics[: 1, :], dynamics[:, 1, :])
        overage = torch.norm(boundary[:, 1, 1] - position[:, 1], dim=-1)
        position[:, 1] = torch.where(overage > torch.zeros(overage.shape),
                                     boundary[:, 1, 1] + overage,
                                     position[:, 1])
        dynamics[:, 1, :] = torch.where(overage > torch.zeros(overage.shape),
                                        -dynamics[: 1, :], dynamics[:, 1, :])

        self.observe('position_%d' % t, data.get('position_%d' % t, None),
                     Normal, loc=position, scale=softplus(noise))

        return boundary, dynamics, uncertainty, noise, position
