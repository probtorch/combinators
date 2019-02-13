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

def reflect_on_boundary(position, dynamics, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = sign * (position[:, d] - boundary)
    soft_overage = softplus(overage)
    positions = list(torch.unbind(position, 1))
    positions[d] = positions[d] + sign * 2 * soft_overage
    position = torch.stack(positions, dim=-1)
    overage = overage.unsqueeze(-1).expand(dynamics[:, d].shape)
    dynamics = list(torch.unbind(dynamics, 1))
    dynamics[d] = torch.where(overage > torch.zeros(overage.shape),
                              -dynamics[d], dynamics[d])
    dynamics = torch.stack(dynamics, dim=1)
    return position, dynamics

class StepBallDynamics(combinators.Primitive):
    def _forward(self, theta, t, data={}):
        boundary, dynamics, uncertainty, noise, position = theta
        t += 1

        position = position + self.sample(
            Normal, torch.bmm(dynamics, position.unsqueeze(-1)).squeeze(-1),
            softplus(uncertainty), name='velocity_%d' % t
        )

        for i in range(2):
            for pos in [True, False]:
                position, dynamics = reflect_on_boundary(position, dynamics,
                                                         boundary, d=i,
                                                         positive=pos)
        position = self.observe('position_%d' % t,
                                data.get('position_%d' % t, None), Normal,
                                loc=position, scale=softplus(noise))

        return boundary, dynamics, uncertainty, noise, position
