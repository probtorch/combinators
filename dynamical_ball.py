#!/usr/bin/env python3

import torch
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators

class InitBallDynamics(combinators.Primitive):
    def _forward(self, data={}):
        # Sample a boundary
        walls = torch.Tensor([[1, 0], [0, -1], [-1, 0], [0, 1]])
        boundary = self.param_sample(LogNormal, name='boundary') * walls
        dynamics = self.param_sample(Normal, name='dynamics')
        uncertainty = self.param_sample(LogNormal, name='uncertainty')
        noise = self.param_sample(LogNormal, name='noise')
        position = self.param_sample(MultivariateNormal, name='position_0')
        return boundary, dynamics, uncertainty, noise, position

class StepBallDynamics(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        boundary, dynamics, uncertainty, noise, position = theta
        t += 1

        position = position + self.sample(
            Normal, torch.matmul(dynamics, position), softplus(uncertainty),
            name='velocity_%d' % t
        )

        overage = torch.norm(position[:, 0] - boundary[:, 0, 0], dim=-1)
        position[:, 0] = torch.where(overage > torch.zeros(overage.shape),
                                     boundary[:, 0, 0] - overage,
                                     position[:, 0])
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
