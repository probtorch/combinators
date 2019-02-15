#!/usr/bin/env python3

import torch
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators

class InitBallDynamics(combinators.Primitive):
    def __init__(self, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
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
        dynamics = self.param_sample(Normal, name='dynamics')
        uncertainty = self.param_sample(LogNormal, name='uncertainty')
        noise = self.param_sample(LogNormal, name='noise')
        pos_params = self.args_vardict()['position_0']
        pos_scale = LowerCholeskyTransform()(pos_params['covariance_matrix'])
        position = self.sample(MultivariateNormal, loc=pos_params['loc'],
                               scale_tril=pos_scale, name='position_0')
        return dynamics, uncertainty, noise, position

def reflect_on_boundary(position, dynamics, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = position[:, d] - sign * boundary
    overage = torch.where(torch.sign(overage) == sign, overage,
                          torch.zeros(*overage.shape))
    position = list(torch.unbind(position, 1))
    position[d] = position[d] - overage
    position = torch.stack(position, dim=1)

    overage = overage.unsqueeze(-1).expand(dynamics[:, d].shape)
    dynamics = list(torch.unbind(dynamics, 1))
    dynamics[d] = torch.where(overage != 0.0, -dynamics[d], dynamics[d])
    dynamics = torch.stack(dynamics, dim=1)
    return position, dynamics

class StepBallDynamics(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        dynamics, uncertainty, noise, position = theta

        for i in range(2):
            for pos in [True, False]:
                position, dynamics = reflect_on_boundary(position, dynamics,
                                                         6.0, d=i,
                                                         positive=pos)

        position = position + self.sample(
            Normal, torch.bmm(dynamics, position.unsqueeze(-1)).squeeze(-1),
            softplus(uncertainty), name='velocity_%d' % t
        )
        self.observe('position_%d' % (t+1),
                     data.get('position_%d' % (t+1), None), Normal,
                     loc=position, scale=softplus(noise))

        return dynamics, uncertainty, noise, position

class StepBallGuide(combinators.Primitive):
    def __init__(self, T, params={}, trainable=False, batch_shape=(1,), q=None):
        params = {
            'velocities': {
                'loc': torch.zeros(T, 2),
                'scale': torch.ones(T, 2),
            }
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
        params = self.args_vardict()['velocities']

        self.sample(Normal, params['loc'][:, t],
                    softplus(params['scale'][:, t]), name='velocity_%d' % t)
        return theta
