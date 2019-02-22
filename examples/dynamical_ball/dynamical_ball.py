#!/usr/bin/env python3

import numpy as np
import torch
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators.model

class InitBallDynamics(combinators.model.Primitive):
    def __init__(self, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'direction': {
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
        direction = self.param_sample(Normal, name='direction')
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
                          torch.zeros(*overage.shape))
    position = list(torch.unbind(position, 1))
    position[d] = position[d] - 2 * overage
    position = torch.stack(position, dim=1)

    direction = list(torch.unbind(direction, 1))
    direction[d] = torch.where(overage != 0.0, -direction[d], direction[d])
    direction = torch.stack(direction, dim=1)
    return position, direction, overage

class StepBallDynamics(combinators.model.Primitive):
    def __init__(self, *args, **kwargs):
        super(StepBallDynamics, self).__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss(reduction='none')

    def _forward(self, theta, t, data={}):
        direction, position, uncertainty = theta

        proposal = position + direction

        for i in range(2):
            for pos in [True, False]:
                proposal, direction, overage = reflect_on_boundary(
                    proposal, direction, 6.0, d=i, positive=pos
                )
                self.p.loss(self.loss, overage, torch.zeros(*overage.shape),
                            name='overage_%d_%d_%s' % (t, i, pos))
        self.sample(Normal, loc=proposal - position, scale=uncertainty,
                    name='velocity_%d' % t)
        position = self.observe('position_%d' % (t+1),
                                data.get('position_%d' % (t+1), None),
                                Normal, loc=proposal,
                                scale=torch.ones(*proposal.shape))

        return direction, position, uncertainty

class StepBallGuide(combinators.model.Primitive):
    def __init__(self, T, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'velocities': {
                'loc': torch.zeros(T, 2),
                'scale': torch.ones(T, 2),
            },
        } if not params else params
        self._num_timesteps = T
        super(StepBallGuide, self).__init__(params, trainable, batch_shape, q)

    @property
    def name(self):
        return 'StepBallDynamics'

    def cond(self, qs):
        return StepBallGuide(self._num_timesteps, self.args_vardict(False),
                             self._hyperparams_trainable, self.batch_shape,
                             qs[self.name])

    def _forward(self, theta, t, data={}):
        params = self.args_vardict()
        velocities = params['velocities']
        self.sample(Normal, velocities['loc'][:, t],
                    softplus(velocities['scale'][:, t]), name='velocity_%d' % t)
        return theta
