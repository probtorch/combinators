#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import Normal, OneHotCategorical, RelaxedOneHotCategorical
import torch.nn as nn
from torch.nn.functional import softplus

import combinators.model as model

class AiStatePrior(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 1)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim) * 0.05,
                },
                'observation_noise': {
                    'loc': torch.ones(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                }
            }
        super(AiStatePrior, self).__init__(*args, **kwargs)

    def _forward(self, env=None):
        state = self.param_sample(Normal, name='state_0')

        observation, _, _, _ = env.retrieve_step(0, None, True)
        observation = torch.Tensor(observation).to(state.device).expand(
            self.batch_shape + observation.shape
        )
        observation_noise = self.param_sample(Normal, 'observation_noise')

        state = self.observe('observation_0', observation, Normal, loc=state,
                             scale=softplus(observation_noise))
        return state

class CartpoleStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 1)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 1)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'costs': {
                    'loc': torch.zeros(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
                'control': {
                    'temperature': torch.ones(1),
                    'probs': torch.ones(self._action_dim),
                },
                'observation_noise': {
                    'loc': torch.ones(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
            }
            kwargs['params']['costs']['scale'][0] = 2.4 / 2
            kwargs['params']['costs']['scale'][2] = np.pi / (15 * 2)
        super(CartpoleStep, self).__init__(*args, **kwargs)
        self.propagate_state = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, 8),
            nn.Softsign(),
            nn.Linear(8, self._state_dim * 2),
        )

    def _forward(self, theta, t, env=None):
        prev_state = theta

        control = self.param_sample(RelaxedOneHotCategorical, name='control')
        control = self.sample(OneHotCategorical, probs=control, name='action')

        state = self.propagate_state(torch.cat((prev_state, control), dim=-1))
        state = state.reshape(-1, self._state_dim, 2)
        state = self.sample(Normal, state[:, :, 0], softplus(state[:, :, 1]),
                            name='state_%d' % (t+1))

        observation, _, _, _ = env.retrieve_step(
            t + 1, control.argmax(dim=-1)[0].item(), override_done=True
        )
        if observation is not None:
            observation = torch.Tensor(observation).to(state.device).expand(
                self.batch_shape + observation.shape
            )
        observation_noise = self.param_sample(Normal, 'observation_noise')
        state = self.observe('observation_%d' % (t+1), observation, Normal,
                             state, softplus(observation_noise))
        self.param_observe(Normal, 'costs', state)

        return state

class AiStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 1)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 1)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'observation_noise': {
                    'loc': torch.ones(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
            }
        super(AiStep, self).__init__(*args, **kwargs)
        self.decode_policy = nn.Sequential(
            nn.Linear(self._state_dim, 8),
            nn.Softsign(),
            nn.Linear(8, self._action_dim + 1),
        )
        self.propagate_state = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, 8),
            nn.Softsign(),
            nn.Linear(8, self._state_dim * 2),
        )

    @property
    def name(self):
        return 'CartpoleStep'

    def _forward(self, theta, t, env=None):
        prev_state = theta

        control = self.decode_policy(prev_state)
        control = self.sample(RelaxedOneHotCategorical,
                              temperature=softplus(control[:, :1]),
                              probs=softplus(control[:, 1:]), name='control')
        control = self.sample(OneHotCategorical, probs=control, name='action')

        state = self.propagate_state(torch.cat((prev_state, control), dim=-1))
        state = state.reshape(-1, self._state_dim, 2)
        state = self.sample(Normal, loc=state[:, :, 0],
                            scale=softplus(state[:, :, 1]),
                            name='state_%d' % (t+1))

        self.param_sample(Normal, 'observation_noise')

        return state
