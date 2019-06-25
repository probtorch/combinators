#!/usr/bin/env python3

import gym
import logging
import torch

from .. import graphs
from ..inference import importance

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

def active_inference_episode(agent, env, episode_length=10, dream=True,
                             render=False, finish=False):
    last_iteration = episode_length
    t = 0
    control = None
    prediction = None
    observation = None
    graph = graphs.ComputationGraph()

    param = list(agent.parameters())[0]
    episode_log_weight = torch.zeros(*agent.batch_shape).to(param)
    while t < episode_length and not (env.done and finish):
        if render:
            env.render()
        if control is not None:
            action = control[0].cpu().detach().numpy()
        else:
            action = None
        observation, reward, done, _ = env.retrieve_step(t, action)
        if observation is not None:
            observation = torch.Tensor(observation).to(episode_log_weight)
            reward = torch.Tensor([reward]).to(episode_log_weight)
            obs_done = torch.eye(2)[1 if done else 0].to(episode_log_weight)
            observation = torch.cat((observation, reward, obs_done), dim=-1)
            observation = observation.expand(*agent.batch_shape,
                                             *observation.shape)
        (control, prediction), step_graph, step_log_weight = agent(control,
                                                                   prediction,
                                                                   observation)
        t += 1
        if not dream:
            env.focus(t)
        graph.insert(str(t), step_graph)

        if env.done and last_iteration == episode_length:
            last_iteration = t

        episode_log_weight = episode_log_weight.to(device=graph.device)
        episode_log_weight = episode_log_weight + step_log_weight

    return (control, prediction), episode_log_weight, last_iteration, graph

def active_inference(agent, env_name, lr=1e-6, episode_length=10, use_cuda=True,
                     episodes=1, dream=True, patience=None):
    agent.train()
    if torch.cuda.is_available() and use_cuda:
        agent.cuda()
    env = AiGymEnv(gym.make(env_name))

    optimizer = torch.optim.Adam(list(agent.parameters()), lr=lr)
    if patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=1e-6, patience=patience, verbose=True,
            mode='max', cooldown=patience/2,
        )

    for episode in range(episodes):
        focus = 1
        env.reset()

        if dream:
            while not env.done and focus < episode_length:
                optimizer.zero_grad()

                zs, log_weight, length, graph = active_inference_episode(
                    agent, env, episode_length=episode_length, dream=True,
                )

                elbo = importance.elbo(log_weight, iwae_objective=True,
                                       xi=graph)
                (-elbo).backward()
                optimizer.step()
                if patience:
                    scheduler.step(elbo)

                focus += 1
                env.focus(focus)
        else:
            optimizer.zero_grad()

            zs, log_weight, length, graph = active_inference_episode(
                agent, env, episode_length=episode_length, dream=False,
            )

            elbo = importance.elbo(log_weight, iwae_objective=True, xi=graph) / len(graph)
            (-elbo).backward()
            optimizer.step()
            if patience:
                scheduler.step(elbo)
        logging.info('ELBO=%.8e at episode %d of length %d', elbo, episode,
                     length)
        env.close()

    if torch.cuda.is_available() and use_cuda:
        agent.cpu()
        torch.cuda.empty_cache()
    agent.eval()

    return zs[:-1], graph, log_weight

def active_inference_test(agent, env_name, use_cuda=True, iterations=200,
                          online_inference=True, lr=1e-4):
    if torch.cuda.is_available() and use_cuda:
        agent.cuda()
    env = AiGymEnv(gym.make(env_name))
    graph = graphs.ComputationGraph()
    env.reset()
    env.render()

    if online_inference:
        optimizer = torch.optim.Adam(list(agent.parameters()), lr=lr)
        focus = 1
        while not env.done and focus < iterations:
            optimizer.zero_grad()

            zs, log_weight, _, egraph = active_inference_episode(
                agent, env, episode_length=iterations, dream=True, render=False,
                finish=True
            )

            elbo = importance.elbo(log_weight, iwae_objective=True,
                                   xi=egraph)
            (-elbo).backward()
            optimizer.step()
            focus += 1
            env.focus(focus)
            env.render()
    else:
        zs, log_weight, _, egraph = active_inference_episode(
            agent, env, episode_length=iterations, dream=False, render=True,
            finish=True
        )

    graph.insert('0', egraph)
    env.close()

    if torch.cuda.is_available() and use_cuda:
        agent.cpu()
        torch.cuda.empty_cache()

    return zs, graph, log_weight
