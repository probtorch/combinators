#!/usr/bin/env python3

import numpy as np
import torch
from torch.distributions import Beta, Categorical, Dirichlet, MultivariateNormal
from torch.distributions import Normal
from torch.distributions.transforms import LowerCholeskyTransform

import utils

REFLECTIONS = torch.tensor([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]],
                           requires_grad=False)

def reflect_directions(angle):
    unsqueeze = len(angle.shape) < 1
    if unsqueeze:
        angle = angle.unsqueeze(0)
    unit = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    result = REFLECTIONS.to(unit).unsqueeze(0) * unit.unsqueeze(1)
    if unsqueeze:
        result = result.squeeze(0)
    return result

def init_bouncing_ball(params=None, trace=None, data={}):
    initial_alpha = trace.param_sample(Dirichlet, params, name='alpha_0')
    transition_alpha = torch.stack([
        trace.param_sample(Dirichlet, params, name='alpha_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    initial_position = trace.param_sample(Normal, params, name='position_0')
    pi = trace.sample(Dirichlet, initial_alpha, name='Pi')
    initial_z = trace.variable(Categorical, pi, name='direction_0')
    transition = torch.stack([
        trace.sample(Dirichlet, transition_alpha[:, d], name='A_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    dir_angle = trace.param_sample(Beta, params, name='directions__angle') * np.pi/2
    dir_locs = reflect_directions(dir_angle)
    dir_covs = trace.param_sample(Normal, params, name='directions__scale')

    return initial_position, initial_z, transition, dir_locs, dir_covs

def bouncing_ball_step(theta, t, trace=None, data={}):
    position, z_prev, transition, dir_locs, dir_covs = theta
    directions = {
        'loc': dir_locs,
        'covariance_matrix': dir_covs,
    }
    t += 1

    transition_prev = utils.particle_index(transition, z_prev)
    z_current = trace.variable(Categorical, transition_prev,
                               name='direction_%d' % t)
    direction = utils.vardict_particle_index(directions, z_current)
    direction_covariance = direction['covariance_matrix']
    velocity = trace.observe(
        MultivariateNormal, data.get('displacement_%d' % t), direction['loc'],
        scale_tril=LowerCholeskyTransform()(direction_covariance),
        name='displacement_%d' % t,
    )
    position = position + velocity

    return position, z_current, transition, dir_locs, dir_covs
