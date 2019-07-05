#!/usr/bin/env python3

from contextlib import contextmanager
import gym
import logging
import torch

from .. import graphs
from ..inference import importance
from .model import Model
from ..sampler import Sampler

class ActiveSimulation(Model):
    def __init__(self, agent, horizon=5):
        assert isinstance(agent, Sampler)
        super(ActiveSimulation, self).__init__(batch_shape=agent.batch_shape)
        self.add_module('agent', agent)
        self._horizon = horizon

    @property
    def horizon(self):
        return self._horizon

    @contextmanager
    def cond(self, qs):
        with self.agent.cond(qs[self.name:]) as self.agent:
            yield self

    @contextmanager
    def weight_target(self, weights=None):
        with self.agent.weight_target(weights) as self.agent:
            yield self

    @property
    def name(self):
        return 'ActiveSimulation'

    def walk(self, f):
        walk_agent = self.agent.walk(f)
        return f(ActiveSimulation(walk_agent))

    def forward(self, prediction, control, observation):
        graph = graphs.ComputationGraph()
        log_weight = torch.zeros(self.batch_shape).to(control)

        predictions = [{} for _ in range(self.horizon + 1)]
        predictions[0] = prediction
        controls = [torch.zeros(self.batch_shape).to(control)
                    for _ in range(self.horizon + 1)]
        controls[0] = control
        observations = [None for _ in range(self.horizon)]
        observations[0] = observation
        sim_graphs = [None for _ in range(self.horizon)]

        for i in range(self.horizon):
            obs = observation if i == 0 else None
            (controls[i+1], predictions[i+1]), sim_graphs[i], log_weight_i =\
                self.agent(predictions[i], controls[i], obs)
            log_weight = log_weight.to(log_weight_i) + log_weight_i

        graph.insert(self.name, sim_graphs[0])
        return (controls[1], predictions[1]), graph, log_weight

class ActiveEpisode(Model):
    def __init__(self, agent, env_name, target_weights=None,
                 max_episode_length=2000):
        assert isinstance(agent, Sampler)
        super(ActiveEpisode, self).__init__(batch_shape=agent.batch_shape)
        self.add_module('agent', agent)
        self._env_name = env_name
        self._env = gym.make(env_name)
        self._max_episode_length = max_episode_length
        self._qs = None
        self._target_weights = target_weights

    @contextmanager
    def cond(self, qs):
        original_qs = self._qs
        try:
            self._qs = qs
            yield self
        finally:
            self._qs = original_qs

    @contextmanager
    def weight_target(self, weights=None):
        original_target_weights = self._target_weights
        try:
            self._target_weights = weights
            yield self
        finally:
            self._target_weights = original_target_weights

    @contextmanager
    def _ready(self, t):
        with self.agent.weight_target(self._target_weights) as agent:
            original_agent = self.agent
            self.agent = agent
            if self._qs and self._qs.contains_model(self.name + '/' + str(t)):
                with agent.cond(self._qs[self.name + '/' + str(t):]) as aq:
                    self.agent = aq
                    yield self
            else:
                yield self
            self.agent = original_agent

    @property
    def name(self):
        return 'ActiveEpisode(%s)' % self._env_name

    def walk(self, f):
        walk_agent = self.agent.walk(f)
        return f(ActiveEpisode(walk_agent, self._env))

    def forward(self, render=False, data={}):
        graph = graphs.ComputationGraph()
        log_weight = torch.zeros(self.agent.batch_shape).to(
            list(self.parameters())[0]
        )

        t = 0
        dynamics = None
        prediction = None
        control = None
        observation = torch.Tensor(list(self._env.reset()) + [0.])
        observation = observation.to(log_weight).expand(*self.batch_shape,
                                                        *observation.shape)
        done = False
        final = False

        while not done or final:
            if render:
                self._env.render()
            with self._ready(t) as _:
                (dynamics, control, prediction), graph_t, log_weight_t =\
                    self.agent(dynamics, control, prediction, observation)
            if control.dtype == torch.int:
                action = control[0].argmax(dim=-1).item()
            else:
                action = control[0].cpu().detach().numpy()

            if self._qs and self._qs.contains_model(self.name + '/' + str(t)):
                agent_name = self.name + '/' + str(t) + '/' + self.agent.name
                next_name = self.name + '/' + str(t + 1)
                done = not self._qs.contains_model(next_name)
                observation = self._qs[agent_name]['observation'].value
            else:
                observation, reward, done, _ = self._env.step(action)
                observation = torch.Tensor(observation).expand(
                    *self.batch_shape, *observation.shape
                ).to(log_weight)
                reward = torch.tensor([reward]).expand(*self.batch_shape, 1)
                reward = reward.to(observation)
                observation = torch.cat((observation, reward), dim=-1)
            final = done if not final else False

            graph.insert(self.name + '/' + str(t), graph_t)
            log_weight = log_weight.to(device=graph.device) + log_weight_t
            t += 1

        self._env.close()
        if not render and not self._qs:
            logging.info('Episode length: %d', t)
        return (control, prediction, t), graph, log_weight

def active_logger(objectives, t, xi=None):
    elbo = -objectives[0] / len(xi)
    logging.info('ELBO=%.8e per step at epoch %d', elbo, t + 1)
    return [elbo]

def active_variational(episode, num_iterations, use_cuda=True, lr=1e-3,
                       log_estimator=False, patience=10):
    active_elbo = {
        'name': 'elbo',
        'function': lambda log_weight, xi=None: -importance.elbo(
            log_weight, iwae_objective=log_estimator, xi=xi
        ),
    }
    param_groups = [{
        'objective': active_elbo,
        'optimizer_args': {
            'params': episode.parameters(),
            'lr': lr,
        },
        'patience': patience,
    }]

    return importance.multiobjective_variational(episode, param_groups,
                                                 num_iterations, {}, use_cuda,
                                                 logger=active_logger)
