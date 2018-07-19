#!/usr/bin/env python3

import torch
from torch.nn.functional import softplus

import utils

def init_bouncing_ball(this=None):
    params = this.args_vardict()

    initial_position = this.trace.param_normal(params, name='position_0')
    pi0 = this.trace.param_dirichlet(params, name='Pi_0')
    initial_z = utils.relaxed_categorical(pi0, 'direction_0', this)
    pi = torch.stack([this.trace.param_dirichlet(params, name='Pi_%d' % (d+1))
                      for d in range(4)], dim=-1)
    initial_speed = this.trace.param_log_normal(params, name='speed')

    doubt = this.trace.param_log_normal(params, name='doubt')
    noise = this.trace.param_log_normal(params, name='noise')

    return initial_position, initial_speed, initial_z, doubt, noise, pi

def bouncing_ball_step(theta, t, this=None):
    prev_position, speed, z_prev, doubt, noise, pi = theta
    params = this.args_vardict()
    t += 1

    pi_prev = utils.relaxed_particle_index(pi, z_prev, this=this)
    direction, z_current = utils.relaxed_index_select(params['directions'],
                                                      pi_prev,
                                                      'direction_%d' % t,
                                                      this=this)
    velocity = speed * direction
    position = this.trace.normal(prev_position + velocity * this.delta_t,
                                 softplus(doubt),
                                 name='position_%d' % t)

    # Jitter observation with noise on top of doubt
    this.trace.normal(
        position - prev_position, softplus(noise), name='displacement_%d' % t,
        value=utils.optional_to(this.guide['displacement_%d' % t], position)
    )

    return position, speed, z_current, doubt, noise, pi
