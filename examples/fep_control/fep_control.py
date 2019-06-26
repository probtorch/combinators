#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import Bernoulli, Beta, Normal
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from torch.distributions.transforms import LowerCholeskyTransform
import torch.nn as nn
from torch.nn.functional import hardtanh, softplus

import combinators.model as model

class NormalEnergy(nn.Module):
    def __init__(self, loc, scale):
        super(NormalEnergy, self).__init__()
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)
        self.all_steps = False

    def forward(self, agent, observation):
        return agent.observe('goal', observation, Normal, self.loc, self.scale)

class BoundedRewardEnergy(nn.Module):
    def __init__(self):
        super(BoundedRewardEnergy, self).__init__()
        self.register_parameter('lower_loc', nn.Parameter(torch.zeros(1)))
        self.register_parameter('lower_scale', nn.Parameter(torch.ones(1)*0.5))
        self.register_parameter('upper_loc', nn.Parameter(torch.ones(1)))
        self.register_parameter('upper_scale', nn.Parameter(torch.ones(1)*0.5))
        self.register_parameter('alpha', nn.Parameter(torch.ones(1)))
        self.register_parameter('beta', nn.Parameter(torch.ones(1)))
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.all_steps = True

    def forward(self, agent, observation):
        lower_bound = agent.sample(Normal, self.lower_loc,
                                   softplus(self.lower_scale),
                                   name='reward_lower_bound')
        upper_bound = agent.sample(Normal, self.upper_loc,
                                   softplus(self.upper_scale),
                                   name='reward_upper_bound')
        normalized_reward = agent.sample(Beta, self.alpha, self.beta,
                                         name='normalized_reward')
        optimality = torch.ones_like(normalized_reward)
        reward_range = upper_bound - lower_bound
        expected_reward = normalized_reward * reward_range + lower_bound
        agent.observe('goal', optimality, Bernoulli, normalized_reward)
        agent.observe('reward', observation[:, -1], Normal, expected_reward,
                      self.scale)

class GenerativeAgent(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        goal = kwargs.pop('goal')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
            }
            if self._discrete_actions:
                kwargs['params']['control'] = {
                    'probs': torch.ones(self._action_dim)
                }
            else:
                kwargs['params']['control'] = {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
        super(GenerativeAgent, self).__init__(*args, **kwargs)
        self.goal = goal
        self.state_transition = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim * 2),
        )
        self.predict_observation = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, (self._observation_dim + 1) * 2)
        )

    def _forward(self, prev_control=None, prediction=None, observation=None):
        if prediction is None:
            state = self.param_sample(Normal, 'state')
        else:
            state = self.sample(Normal, **prediction, name='state')
        if prev_control is None:
            prev_control = torch.zeros(self._action_dim).to(state)

        observable = self.predict_observation(state).reshape(
            -1, self._observation_dim + 1, 2
        )
        observation = self.observe('observation', observation, Normal,
                                   observable[:, :, 0],
                                   softplus(observable[:, :, 1]))
        self.goal(self, observation)

        if self._discrete_actions:
            control = self.param_sample(OneHotCategorical, name='control')
        else:
            control = self.param_sample(Normal, 'control')

        dynamics = self.state_transition(torch.cat((state, control), dim=-1))
        dynamics = dynamics.reshape(-1, self._state_dim, 2)
        prediction = {
            'loc': dynamics[:, :, 0],
            'scale': softplus(dynamics[:, :, 1]),
        }

        return control, prediction

class RecognitionAgent(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        self._name = kwargs.pop('name')
        super(RecognitionAgent, self).__init__(*args, **kwargs)
        self.decode_policy = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._action_dim * 16),
            nn.PReLU(),
            nn.Linear(self._action_dim * 16, self._action_dim * 2),
            nn.Softmax(dim=-1) if self._discrete_actions else nn.Identity(),
        )
        self.encode_state = nn.Sequential(
            nn.Linear(self._observation_dim + 1, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim * 2),
        )

    @property
    def name(self):
        return self._name

    def _forward(self, prev_control=None, prediction=None, observation=None):
        if observation is not None:
            state = self.encode_state(observation).reshape(-1, self._state_dim,
                                                           2)
            prediction = {
                'loc': state[:, :, 0],
                'scale': state[:, :, 1],
            }
        state = self.sample(Normal, prediction['loc'],
                            softplus(prediction['scale']) + 1e-9, name='state')
        if prev_control is None:
            prev_control = torch.zeros(*self.batch_shape, self._action_dim).to(
                state
            )

        control = self.decode_policy(state)
        if self._discrete_actions:
            control = self.sample(OneHotCategorical, probs=control,
                                  name='control')
        else:
            control = control.reshape(-1, self._action_dim, 2)
            if not self.q and observation is not None:
                action = torch.normal(
                    hardtanh(prev_control[0] + control[0, :, 0]),
                    softplus(control[0, :, 1]) + 1e-9
                )
                action = action.expand(*self.batch_shape, self._action_dim)
            elif self.q:
                action = self.q['control'].value
            else:
                action = None
            control = self.sample(Normal,
                                  hardtanh(prev_control + control[:, :, 0]),
                                  softplus(control[:, :, 1]) + 1e-9,
                                  value=action, name='control')

class MountainCarEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.tensor([0.45]).expand(*batch_shape, 1)
        scale = torch.tensor([0.05]).expand(*batch_shape, 1)
        super(MountainCarEnergy, self).__init__(loc, scale)

    def forward(self, agent, observation):
        return super(MountainCarEnergy, self).forward(agent, observation[:, 0])

class MountainCarAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['action_dim'] = 1
        kwargs['observation_dim'] = 2
        kwargs['goal'] = MountainCarEnergy(kwargs['batch_shape'])
        super(MountainCarAgent, self).__init__(*args, **kwargs)

class CartpoleEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.zeros(*batch_shape, 1)
        scale = torch.tensor([np.pi / (15 * 2)]).expand(*batch_shape, 1)
        super(CartpoleEnergy, self).__init__(loc, scale)
        self.all_steps = True

    def forward(self, agent, observation):
        return super(CartpoleEnergy, self).forward(agent, observation)

class CartpoleAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = True
        kwargs['observation_dim'] = 4
        kwargs['goal'] = CartpoleEnergy(kwargs['batch_shape'])
        super(CartpoleAgent, self).__init__(*args, **kwargs)

class BipedalWalkerEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.tensor([0., 0., 1., 0.]).expand(*batch_shape, 4)
        scale = torch.tensor([0.25, 1., 0.0025, 0.0025]).expand(*batch_shape, 4)
        super(BipedalWalkerEnergy, self).__init__(loc, scale)
        self.all_steps = True

    def forward(self, agent, observation):
        return super(BipedalWalkerEnergy, self).forward(agent,
                                                        observation[:, 0:4])

class BipedalWalkerAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['observation_dim'] = 24
        kwargs['action_dim'] = 4
        kwargs['goal'] = BipedalWalkerEnergy(kwargs['batch_shape'])
        super(BipedalWalkerAgent, self).__init__(*args, **kwargs)
