#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import Bernoulli, MultivariateNormal, Normal
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from torch.distributions.transforms import LowerCholeskyTransform
import torch.nn as nn
from torch.nn.functional import softplus

import combinators.model as model

class NormalInterval(nn.Module):
    def __init__(self, loc, scale, num_scales):
        super(NormalInterval, self).__init__()
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)
        self.num_scales = num_scales
        self.all_steps = False

    def forward(self, observation):
        p = Normal(self.loc, self.scale).cdf(observation)
        p = torch.where(p > 0.5, 1. - p, p)
        return 2 * p, observation

class GenerativeActor(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        goal = kwargs.pop('goal')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'state_uncertainty': {
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
        super(GenerativeActor, self).__init__(*args, **kwargs)
        self.goal = goal
        self.state_transition = nn.Sequential(
            nn.Linear(self._state_dim * 2 + self._action_dim,
                      self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim),
        )
        self.predict_observation = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._observation_dim),
        )

    def _forward(self, theta, t, env=None):
        if theta is None:
            state = self.param_sample(Normal, 'state_0')
            if self._discrete_actions:
                control = self.param_sample(OneHotCategorical, name='control')
            else:
                control = self.param_sample(Normal, 'control')
        else:
            prev_state, prev_control = theta
            state_uncertainty = self.param_sample(Normal,
                                                  name='state_uncertainty')

            if self._discrete_actions:
                control = self.param_sample(OneHotCategorical, name='control')
            else:
                control = self.param_sample(Normal, name='control')

            state = self.state_transition(
                torch.cat((prev_state, control, state_uncertainty), dim=-1)
            )

        prediction = self.predict_observation(state)
        if not env.done or self.goal.all_steps:
            goal_prob, comparator = self.goal(prediction)
            self.observe('goal', torch.ones_like(comparator), Bernoulli,
                         probs=goal_prob)

        return state, control, prediction, t, env

class GenerativeObserver(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._observation_dim = kwargs.pop('observation_dim')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
            }
        super(GenerativeObserver, self).__init__(*args, **kwargs)

    def _forward(self, state, control, prediction, t, env=None):
        if isinstance(control, torch.Tensor):
            action = torch.tanh(control[0]).cpu().detach().numpy()
        else:
            action = control
        observation, _, _, _ = env.retrieve_step(t, action, override_done=True)
        if observation is not None:
            observation = torch.Tensor(observation).to(state).expand(
                self.batch_shape + observation.shape
            )

        observation_noise = self.param_sample(Normal,
                                              name='observation_noise')
        observation_scale = LowerCholeskyTransform()(observation_noise)
        if observation is not None:
            self.observe('observation', observation, MultivariateNormal,
                         prediction, scale_tril=observation_scale)

        return state, control

class RecognitionActor(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        self._name = kwargs.pop('name')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'control': {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
            }
        super(RecognitionActor, self).__init__(*args, **kwargs)
        self.decode_policy = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim,
                      self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._action_dim * 16),
            nn.PReLU(),
            nn.Linear(self._action_dim * 16, self._action_dim * 2),
            nn.Softmax(dim=-1) if self._discrete_actions else nn.Tanh(),
        )

    @property
    def name(self):
        return self._name

    def _forward(self, theta, t, env=None):
        if theta is None:
            prev_state = self.param_sample(Normal, 'state_0')
            control = self.param_sample(Normal, 'control')
        else:
            prev_state, prev_control = theta

            control = self.decode_policy(
                torch.cat((prev_state, prev_control), dim=-1)
            )
            if self._discrete_actions:
                control = self.sample(OneHotCategorical, probs=control,
                                      name='control')
            else:
                control = control.reshape(-1, self._action_dim, 2)
                control = prev_control + self.sample(Normal, control[:, :, 0],
                                                     softplus(control[:, :, 1]),
                                                     name='control')
        return prev_state, control, t, env

class RecognitionEncoder(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
            }
        super(RecognitionEncoder, self).__init__(*args, **kwargs)
        self.encode_uncertainty = nn.Sequential(
            nn.Linear(self._observation_dim + self._state_dim,
                      self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim * 2),
        )

    @property
    def name(self):
        return 'GenerativeObserver'

    def _forward(self, prev_state, control, t, env=None):
        if isinstance(control, torch.Tensor):
            action = torch.tanh(control[0]).cpu().detach().numpy()
        else:
            action = control
        observation, _, _, _ = env.retrieve_step(t, action, override_done=True)
        if observation is not None:
            self.param_sample(Normal, name='observation_noise')
            observation = torch.Tensor(observation).to(control).expand(
                self.batch_shape + observation.shape
            )
            state_uncertainty = self.encode_uncertainty(torch.cat(
                (observation, prev_state), dim=-1
            )).reshape(-1, self._state_dim, 2)
            self.sample(Normal, state_uncertainty[:, :, 0],
                        softplus(state_uncertainty[:, :, 1]),
                        name='state_uncertainty')

class MountainCarInterval(NormalInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0.5]).expand(*batch_shape, 1)
        scale = torch.tensor([0.05]).expand(*batch_shape, 1)
        super(MountainCarInterval, self).__init__(loc, scale, 1)

    def forward(self, observation):
        p, _ = super(MountainCarInterval, self).forward(observation[:, 0])
        return p, observation[:, 0]

class MountainCarActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['action_dim'] = 1
        kwargs['observation_dim'] = 2
        kwargs['goal'] = MountainCarInterval(kwargs['batch_shape'])
        super(MountainCarActor, self).__init__(*args, **kwargs)

class CartpoleInterval(NormalInterval):
    def __init__(self, batch_shape):
        loc = torch.zeros(*batch_shape, 1)
        scale = torch.tensor([np.pi / (15 * 2)]).expand(*batch_shape, 1)
        super(CartpoleInterval, self).__init__(loc, scale, 1)
        self.all_steps = True

    def forward(self, observation):
        return super(CartpoleInterval, self).forward(observation)

class CartpoleActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = True
        kwargs['observation_dim'] = 4
        kwargs['goal'] = CartpoleInterval(kwargs['batch_shape'])
        super(CartpoleActor, self).__init__(*args, **kwargs)

class BipedalWalkerInterval(NormalInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0., 0., 1.]).expand(*batch_shape, 3)
        scale = torch.ones(*batch_shape, 3) * 0.0025
        super(BipedalWalkerInterval, self).__init__(loc, scale, 1)
        self.all_steps = True

    def forward(self, observation):
        p, _ = super(BipedalWalkerInterval, self).forward(observation[:, 0:3])
        return p, observation[:, 0:3]

class BipedalWalkerActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['observation_dim'] = 24
        kwargs['action_dim'] = 4
        kwargs['goal'] = BipedalWalkerInterval(kwargs['batch_shape'])
        super(BipedalWalkerActor, self).__init__(*args, **kwargs)
