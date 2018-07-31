#!/usr/bin/env python3

import torch
from torch.nn.functional import softplus

import utils

def init_bouncing_ball(this=None):
    params = this.args_vardict()

    initial_alpha = this.trace.param_dirichlet(params, name='alpha_0')
    transition_alpha = torch.stack([
        this.trace.param_dirichlet(params, name='alpha_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    initial_position = this.trace.param_normal(
        params, name='position_0',
        value=utils.optional_to(this.guide['position_0'], initial_alpha)
    )
    pi = this.trace.dirichlet(initial_alpha, name='Pi')
    initial_z = utils.relaxed_categorical(pi, 'direction_0', this)
    transition = torch.stack([
        this.trace.dirichlet(transition_alpha[:, d], name='A_%d' % (d+1))
        for d in range(4)
    ], dim=-1)
    dir_locs = this.trace.param_normal(params, name='directions__loc')
    dir_covs = this.trace.param_normal(params, name='directions__scale')

    return initial_position, initial_z, transition, dir_locs, dir_covs

def bouncing_ball_step(theta, t, this=None):
    position, z_prev, transition, dir_locs, dir_covs = theta
    directions = {
        'loc': dir_locs,
        'covariance_matrix': dir_covs,
    }
    t += 1

    transition_prev = utils.relaxed_particle_index(transition, z_prev,
                                                   this=this)
    direction, z_current = utils.relaxed_vardict_index_select(
        directions, transition_prev, 'direction_%d' % t, this=this
    )
    direction_covariance = direction['covariance_matrix']
    direction_covariance @= direction_covariance.transpose(1, 2)
    velocity = this.trace.multivariate_normal(
        direction['loc'], softplus(direction_covariance),
        name='displacement_%d' % t,
        value=utils.optional_to(this.guide['displacement_%d' % t], position)
    )
    position = position + velocity

    return position, z_current, transition, dir_locs, dir_covs
