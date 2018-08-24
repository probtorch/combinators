#!/usr/bin/env python3

import torch
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

def init_bouncing_ball(this=None):
    params = this.args_vardict()

    initial_alpha = this.trace.param_dirichlet(params, name='alpha_0')
    transition_alpha = torch.stack([
        this.trace.param_dirichlet(params, name='alpha_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    initial_position = this.trace.param_normal(params, name='position_0')
    pi = this.trace.dirichlet(initial_alpha, name='Pi')
    initial_z = this.trace.variable(torch.distributions.Categorical, pi,
                                    name='direction_0')
    transition = torch.stack([
        this.trace.dirichlet(transition_alpha[:, d], name='A_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    dir_angle = this.trace.param_beta(params, name='directions__angle')
    dir_locs = reflect_directions(dir_angle)
    dir_covs = this.trace.param_normal(params, name='directions__scale')

    return initial_position, initial_z, transition, dir_locs, dir_covs

def bouncing_ball_step(theta, t, this=None):
    position, z_prev, transition, dir_locs, dir_covs = theta
    directions = {
        'loc': dir_locs,
        'covariance_matrix': dir_covs,
    }
    t += 1

    transition_prev = utils.particle_index(transition, z_prev)
    z_current = this.trace.variable(torch.distributions.Categorical,
                                    transition_prev, name='direction_%d' % t)
    direction = utils.vardict_particle_index(directions, z_current)
    direction_covariance = direction['covariance_matrix']
    velocity = this.trace.multivariate_normal(
        direction['loc'],
        scale_tril=LowerCholeskyTransform()(direction_covariance),
        name='displacement_%d' % t
    )
    position = position + velocity

    return position, z_current, transition, dir_locs, dir_covs
