#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import Bernoulli, MultivariateNormal, Normal
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
        self.predictor = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
        )
        self.predict_observation = nn.Linear(self._state_dim * 16,
                                             (self._observation_dim + 1) * 2)
        self.predict_done = nn.Sequential(
            nn.Linear(self._state_dim * 16, 2),
            nn.Softmax(dim=-1),
        )

    def _forward(self, prev_control=None, prediction=None, observation=None):
        if prediction is None:
            state = self.param_sample(Normal, 'state')
        else:
            state = self.sample(Normal, **prediction, name='state')
        if prev_control is None:
            prev_control = torch.zeros(self._action_dim).to(state)

        predictor = self.predictor(state)
        observable = self.predict_observation(predictor).reshape(
            -1, self._observation_dim + 1, 2
        )
        if observation is not None:
            done = observation[:, -2:]
            observation = observation[:, :-2]
        else:
            done = None
        observation = self.observe('observation', observation, Normal,
                                   observable[:, :, 0],
                                   softplus(observable[:, :, 1]))
        self.observe('done', done, OneHotCategorical,
                     self.predict_done(predictor))
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
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._action_dim * 16),
            nn.PReLU(),
            nn.Linear(self._action_dim * 16, self._action_dim * 2),
            nn.Softmax(dim=-1) if self._discrete_actions else nn.Identity(),
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

            control = self.decode_policy(prev_state)
            if self._discrete_actions:
                control = self.sample(OneHotCategorical, probs=control,
                                      name='control')
            else:
                control = control.reshape(-1, self._action_dim, 2)
                control = self.sample(Normal,
                                      hardtanh(prev_control + control[:, :, 0]),
                                      softplus(control[:, :, 1]),
                                      name='control')

        return control, t, env

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
        self.encode_state = nn.Sequential(
            nn.Linear(self._observation_dim, self._state_dim * 4),
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

    def _forward(self, control, t, env=None):
        if isinstance(control, torch.Tensor):
            action = control[0].cpu().detach().numpy()
        else:
            action = control
        observation, _, _, _ = env.retrieve_step(t, action, override_done=True)
        if observation is not None:
            self.param_sample(Normal, name='observation_noise')
            observation = torch.Tensor(observation).to(control).expand(
                self.batch_shape + observation.shape
            )
            state = self.encode_state(observation).reshape(
                -1, self._state_dim, 2
            )
            self.sample(Normal, state[:, :, 0], softplus(state[:, :, 1]),
                        name='state')

class MountainCarEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.tensor([0.6]).expand(*batch_shape, 1)
        scale = torch.tensor([0.05]).expand(*batch_shape, 1)
        super(MountainCarEnergy, self).__init__(loc, scale)

    def forward(self, agent, observation):
        return super(MountainCarEnergy, self).forward(agent, observation[:, 0])

class MountainCarActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['action_dim'] = 1
        kwargs['observation_dim'] = 2
        kwargs['goal'] = MountainCarEnergy(kwargs['batch_shape'])
        super(MountainCarActor, self).__init__(*args, **kwargs)

class CartpoleEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.zeros(*batch_shape, 1)
        scale = torch.tensor([np.pi / (15 * 2)]).expand(*batch_shape, 1)
        super(CartpoleEnergy, self).__init__(loc, scale)
        self.all_steps = True

    def forward(self, agent, observation):
        return super(CartpoleEnergy, self).forward(agent, observation)

class CartpoleActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = True
        kwargs['observation_dim'] = 4
        kwargs['goal'] = CartpoleEnergy(kwargs['batch_shape'])
        super(CartpoleActor, self).__init__(*args, **kwargs)

class BipedalWalkerEnergy(NormalEnergy):
    def __init__(self, batch_shape):
        loc = torch.tensor([0., 0., 1.]).expand(*batch_shape, 3)
        scale = torch.ones(*batch_shape, 3) * 0.0025
        super(BipedalWalkerEnergy, self).__init__(loc, scale)
        self.all_steps = True

    def forward(self, agent, observation):
        return super(BipedalWalkerEnergy, self).forward(agent, observation[:, 0:3])

class BipedalWalkerActor(GenerativeActor):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['observation_dim'] = 24
        kwargs['action_dim'] = 4
        kwargs['goal'] = BipedalWalkerEnergy(kwargs['batch_shape'])
        super(BipedalWalkerActor, self).__init__(*args, **kwargs)
