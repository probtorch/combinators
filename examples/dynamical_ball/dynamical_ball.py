#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators.model

class InitBallDynamics(combinators.model.Primitive):
    def __init__(self, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'velocity_0': {
                'loc': torch.ones(2) / np.sqrt(2),
                'scale': torch.ones(2),
            },
            'position_0': {
                'loc': torch.ones(2),
                'covariance_matrix': torch.eye(2),
            },
            'uncertainty': {
                'loc': torch.ones(2),
                'scale': torch.ones(2),
            },
            'noise': {
                'loc': torch.ones(2),
                'scale': torch.ones(2),
            },
        } if not params else params
        super(InitBallDynamics, self).__init__(params, trainable, batch_shape,
                                               q)

    def _forward(self, data={}):
        direction = self.param_sample(Normal, name='velocity_0')
        speed = torch.sqrt(torch.sum(direction**2, dim=1))
        direction = direction / speed.unsqueeze(-1).expand(*direction.shape)
        pos_params = self.args_vardict()['position_0']
        pos_scale = LowerCholeskyTransform()(pos_params['covariance_matrix'])
        position = self.sample(MultivariateNormal, loc=pos_params['loc'],
                               scale_tril=pos_scale, name='position_0')
        uncertainty = softplus(self.param_sample(Normal, name='uncertainty'))
        noise = softplus(self.param_sample(Normal, name='noise'))
        return direction, position, uncertainty, noise

def reflect_on_boundary(position, direction, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = position[:, d] - sign * boundary
    overage = torch.where(torch.sign(overage) == sign, overage,
                          torch.zeros(*overage.shape, device=position.device))
    position = list(torch.unbind(position, 1))
    position[d] = position[d] - 2 * overage
    position = torch.stack(position, dim=1)

    direction = list(torch.unbind(direction, 1))
    direction[d] = torch.where(overage != 0.0, -direction[d], direction[d])
    direction = torch.stack(direction, dim=1)
    return position, direction

def simulate_step(position, velocity, p=None):
    proposal = position + velocity
    for i in range(2):
        for pos in [True, False]:
            proposal, velocity = reflect_on_boundary(
                proposal, velocity, 6.0, d=i, positive=pos
            )
    return proposal, velocity

def simulate_trajectory(position, velocity, num_steps, velocities=None):
    trajectory = torch.zeros(position.shape[0], num_steps + 1, 2, 2)
    trajectory[:, 0, 0] = position
    trajectory[:, 0, 1] = velocity
    for t in range(1, num_steps + 1):
        if velocities is not None:
            velocity = velocities[:, t-1]
        position, velocity = simulate_step(position, velocity)
        trajectory[:, t, 0] = position
        trajectory[:, t, 1] = velocity
    return trajectory

class StepBallDynamics(combinators.model.Primitive):
    def _forward(self, theta, t, data={}):
        direction, position, uncertainty, noise = theta

        position, direction = simulate_step(position, direction, self.p)
        position = self.observe('position_%d' % (t+1),
                                data.get('position_%d' % (t+1)), Normal,
                                loc=position, scale=noise)
        direction = self.sample(Normal, loc=direction, scale=uncertainty,
                                name='velocity_%d' % (t+1))

        return direction, position, uncertainty, noise

class StepBallGuide(combinators.model.Primitive):
    def __init__(self, num_steps, params={}, trainable=False, batch_shape=(1,),
                 q=None):
        params = {
            'velocities': {
                'loc': torch.zeros(num_steps, 2),
                'scale': torch.ones(num_steps, 2),
            },
        } if not params else params
        super(StepBallGuide, self).__init__(params, trainable, batch_shape, q)
        self._num_steps = num_steps

    @property
    def name(self):
        return 'StepBallDynamics'

    def _forward(self, theta, t, data={}):
        velocities = self.args_vardict()['velocities']

        self.sample(Normal, loc=velocities['loc'][:, t],
                    scale=softplus(velocities['scale'][:, t]),
                    name='velocity_%d' % (t+1))
        return theta
