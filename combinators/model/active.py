#!/usr/bin/env python3

from contextlib import contextmanager
import gym
import logging
import torch

from .. import graphs
from ..inference import importance
from .model import Model
from ..sampler import Sampler

class AiGymEnv:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self._done = False
        self._observations = []
        self._actions = []
        self.focus_time = 0

    def reset(self):
        observation = self.gym_env.reset()
        self._done = False
        self._observations = [(observation, 0.0, False, {})]
        self._actions = []
        self.focus_time = 1
        return observation

    def step(self, action, override_done=False):
        if not self._done or override_done:
            observation, reward, done, info = self.gym_env.step(action)
            self._done = done
        else:
            observation = None
            reward = None
            done = self._done
            info = None
        return observation, reward, done, info

    def render(self, mode='human'):
        return self.gym_env.render(mode=mode)

    def close(self):
        self.gym_env.close()

    @property
    def action_space(self):
        return self.gym_env.action_space

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def done(self):
        return self._done

    def focus(self, t):
        self.focus_time = t

    def retrieve_step(self, t, action, override_done=False):
        if t >= len(self._observations) and t == self.focus_time:
            self._actions.append(action)
            result = self.step(action, override_done)
            self._observations.append(result)
            return self._observations[-1]
        elif t < len(self._observations):
            return self._observations[t]
        return (None, None, self._observations[-1][2], None)

class ActiveEpisode(Model):
    def __init__(self, agent, env_name, target_weights=None,
                 max_episode_length=2000):
        assert isinstance(agent, Sampler)
        super(ActiveEpisode, self).__init__(batch_shape=agent.batch_shape)
        self.add_module('agent', agent)
        self._env_name = env_name
        self._env = AiGymEnv(gym.make(env_name))
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
        t = 0
        prediction = None
        control = None
        observation = None
        done = False
        self._env.reset()

        graph = graphs.ComputationGraph()
        log_weight = torch.zeros(self.agent.batch_shape).to(
            list(self.parameters())[0]
        )
        while not done:
            if control is not None:
                action = control[0].cpu().detach().numpy()
            else:
                action = None
            if self._qs and self._qs.contains_model(self.name + '/' + str(t)):
                agent_name = self.name + '/' + str(t) + '/' + self.agent.name
                next_name = self.name + '/' + str(t + 1)
                done = not self._qs.contains_model(next_name)
                observation = self._qs[agent_name]['observation'].value
            else:
                observation, reward, done, _ = self._env.retrieve_step(
                    t, action, override_done=True
                )
                observation = torch.Tensor(observation).expand(
                    *self.batch_shape, *observation.shape
                ).to(log_weight)
                reward = torch.tensor([reward]).expand(*self.batch_shape, 1)
                reward = reward.to(observation)
                obs_done = torch.eye(2)[1 if done else 0].expand(
                    *self.batch_shape, 2
                ).to(observation)
                observation = torch.cat((observation, reward, obs_done), dim=-1)
            if render:
                self._env.render()
            with self._ready(t) as _:
                (control, prediction), graph_t, log_weight_t = self.agent(
                    control, prediction, observation
                )
            graph.insert(self.name + '/' + str(t), graph_t)
            log_weight = log_weight.to(device=graph.device) + log_weight_t

            t += 1
            self._env.focus(t)

        self._env.close()
        if not render and not self._qs:
            logging.info('Episode length: %d', t)
        return (control, prediction, t), graph, log_weight
