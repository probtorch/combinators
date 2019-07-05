#!/usr/bin/env python3

import numpy as np
import torch
from torch.distributions import Bernoulli, Beta, Normal, OneHotCategorical
from torch.distributions import RelaxedOneHotCategorical, TransformedDistribution
from torch.distributions import Uniform
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import torch.nn as nn
from torch.nn.functional import hardtanh, softplus

import combinators.model as model

class LogisticInterval(nn.Module):
    def __init__(self, loc, scale):
        super(LogisticInterval, self).__init__()
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)

    def forward(self, observation):
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv,
                      AffineTransform(loc=self.loc, scale=self.scale)]
        logistic = TransformedDistribution(base_distribution, transforms)
        p = logistic.cdf(observation)
        p = torch.where(p > 0.5, 1. - p, p)
        return 2 * p

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
        self._dyn_dim = kwargs.pop('dyn_dim', 2)
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2) + 1
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        goal = kwargs.pop('goal')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'dynamics': {
                    'loc': torch.zeros(self._dyn_dim),
                    'scale': torch.ones(self._dyn_dim),
                },
                'state': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
            }
        super(GenerativeAgent, self).__init__(*args, **kwargs)
        self.goal = goal
        self.dynamical_transition = nn.Sequential(
            nn.Linear(self._dyn_dim + self._state_dim + self._action_dim,
                      self._dyn_dim * 2),
            nn.PReLU(),
            nn.Linear(self._dyn_dim * 2, self._dyn_dim * 3),
            nn.PReLU(),
            nn.Linear(self._dyn_dim * 3, self._dyn_dim * 4),
            nn.PReLU(),
            nn.Linear(self._dyn_dim * 4, self._dyn_dim),
        )
        self.project_state = nn.Sequential(
            nn.Linear(self._dyn_dim + self._action_dim, self._state_dim * 2),
            nn.PReLU(),
            nn.Linear(self._state_dim * 2, self._state_dim * 3),
            nn.PReLU(),
            nn.Linear(self._state_dim * 3, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 2),
        )
        self.predict_observation = nn.Sequential(
            nn.Linear(self._dyn_dim + self._state_dim,
                      self._observation_dim * 2),
            nn.PReLU(),
            nn.Linear(self._observation_dim * 2, self._observation_dim * 3),
            nn.PReLU(),
            nn.Linear(self._observation_dim * 3, self._observation_dim * 4),
            nn.PReLU(),
            nn.Linear(self._observation_dim * 4, self._observation_dim * 2)
        )

    def _forward(self, dynamics=None, prev_control=None, prediction=None,
                 observation=None):
        if dynamics is None:
            dynamics = self.param_sample(Normal, 'dynamics')
        if prediction is None:
            state = self.param_sample(Normal, 'state')
        else:
            state = self.sample(Normal, **prediction, name='state')
        if prev_control is None:
            prev_control = torch.zeros(self._action_dim).to(state).expand(
                *self.batch_shape, self._action_dim,
            )

        observable = self.predict_observation(torch.cat((dynamics, state),
                                                        dim=-1))
        observable = observable.reshape(-1, self._observation_dim, 2)
        self.observe('observation', observation, Normal, observable[:, :, 0],
                     softplus(observable[:, :, 1]))
        success = self.goal(observable[:, :, 0])
        success = self.sample(LogitRelaxedBernoulli, torch.ones_like(success),
                              probs=success, name='success')
        self.observe('goal', torch.ones_like(success), Bernoulli,
                     logits=success)

        if self._discrete_actions:
            options = self.sample(RelaxedOneHotCategorical,
                                  torch.ones_like(prev_control),
                                  probs=prev_control)
            control = self.sample(OneHotCategorical, probs=options,
                                  name='control')
        else:
            control = self.sample(Normal, prev_control,
                                  torch.ones_like(prev_control), name='control')
            control = hardtanh(control[0].expand(*control.shape))

        dynamics = self.dynamical_transition(
            torch.cat((dynamics, state, control), dim=-1)
        )
        next_state = self.project_state(torch.cat((dynamics, control), dim=-1))
        next_state = next_state.reshape(-1, self._state_dim, 2)
        prediction = {
            'loc': next_state[:, :, 0],
            'scale': softplus(next_state[:, :, 1]),
        }

        return dynamics, control, prediction

class RecognitionAgent(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._dyn_dim = kwargs.pop('dyn_dim', 2)
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2) + 1
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        self._name = kwargs.pop('name')
        goal = kwargs.pop('goal')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'dynamics': {
                    'loc': torch.zeros(self._dyn_dim),
                    'scale': torch.ones(self._dyn_dim),
                },
            }
        super(RecognitionAgent, self).__init__(*args, **kwargs)
        self.goal = goal
        policy_factor = 1 if self._discrete_actions else 2
        self.encode_policy = nn.Sequential(
            nn.Linear(self._dyn_dim + self._observation_dim,
                      self._action_dim * 2),
            nn.PReLU(),
            nn.Linear(self._action_dim * 2, self._action_dim * 3),
            nn.PReLU(),
            nn.Linear(self._action_dim * 3, self._action_dim * 4),
            nn.PReLU(),
            nn.Linear(self._action_dim * 4, self._action_dim * policy_factor),
            nn.Softmax(dim=-1) if self._discrete_actions else nn.Identity(),
        )
        self.encode_state = nn.Sequential(
            nn.Linear(self._dyn_dim + self._observation_dim,
                      self._state_dim * 2),
            nn.PReLU(),
            nn.Linear(self._state_dim * 2, self._state_dim * 3),
            nn.PReLU(),
            nn.Linear(self._state_dim * 3, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 2),
        )

    @property
    def name(self):
        return self._name

    def _forward(self, dynamics=None, prev_control=None, prediction=None,
                 observation=None):
        if dynamics is None:
            dynamics = self.param_sample(Normal, 'dynamics')
        if prev_control is None:
            prev_control = torch.zeros(*self.batch_shape, self._action_dim).to(
                dynamics
            )
        if observation is not None:
            success = self.goal(observation)
            self.sample(LogitRelaxedBernoulli, torch.ones_like(success),
                        probs=success, name='success')

            observed_information = torch.cat((dynamics, observation), dim=-1)
            control = self.encode_policy(observed_information)
            if self._discrete_actions:
                control = self.sample(OneHotCategorical, probs=control,
                                      name='control')
            else:
                control = control.reshape(-1, self._action_dim, 2)
                control = self.sample(Normal, prev_control + control[:, :, 0],
                                      softplus(control[:, :, 1]),
                                      name='control')

            state = self.encode_state(observed_information).reshape(
                -1, self._state_dim, 2
            )
            prediction = {
                'loc': state[:, :, 0],
                'scale': state[:, :, 1],
            }
        state = self.sample(Normal, prediction['loc'],
                            softplus(prediction['scale']), name='state')

class MountainCarEnergy(LogisticInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0.45]).expand(*batch_shape, 1)
        scale = torch.tensor([0.05]).expand(*batch_shape, 1)
        super(MountainCarEnergy, self).__init__(loc, scale)

    def forward(self, observation):
        return super(MountainCarEnergy, self).forward(observation[:, 0:1])

class MountainCarAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['action_dim'] = 1
        kwargs['observation_dim'] = 2
        kwargs['goal'] = MountainCarEnergy(kwargs['batch_shape'])
        super(MountainCarAgent, self).__init__(*args, **kwargs)

class CartpoleEnergy(LogisticInterval):
    def __init__(self, batch_shape):
        loc = torch.zeros(*batch_shape, 1)
        scale = torch.tensor([np.pi / (15 * 2)]).expand(*batch_shape, 1)
        super(CartpoleEnergy, self).__init__(loc, scale)

    def forward(self, observation):
        return super(CartpoleEnergy, self).forward(observation)

class CartpoleAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = True
        kwargs['observation_dim'] = 4
        kwargs['goal'] = CartpoleEnergy(kwargs['batch_shape'])
        super(CartpoleAgent, self).__init__(*args, **kwargs)

class BipedalWalkerEnergy(LogisticInterval):
    def __init__(self, batch_shape):
        loc = torch.tensor([0., 0., 1., 0.]).expand(*batch_shape, 4)
        scale = torch.tensor([0.25, 1., 0.0025, 0.0025]).expand(*batch_shape, 4)
        super(BipedalWalkerEnergy, self).__init__(loc, scale)

    def forward(self, observation):
        return super(BipedalWalkerEnergy, self).forward(observation[:, 0:4])

class BipedalWalkerAgent(GenerativeAgent):
    def __init__(self, *args, **kwargs):
        kwargs['discrete_actions'] = False
        kwargs['observation_dim'] = 24
        kwargs['action_dim'] = 4
        kwargs['goal'] = BipedalWalkerEnergy(kwargs['batch_shape'])
        super(BipedalWalkerAgent, self).__init__(*args, **kwargs)
