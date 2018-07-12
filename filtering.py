#!/usr/bin/env python3

import collections
import itertools

import probtorch
from probtorch.util import log_sum_exp
import torch

import combinators
import utils

EMPTY_ANNOTATION = collections.defaultdict(lambda: 0.0)

class ForwardMessenger(combinators.Model):
    def __init__(self, f, latent, observation, transition, observation_dists,
                 initial_marginals=None, phi={}, theta={}):
        self._latent = latent
        self._observation = observation
        self.transition = transition
        self.observation_dists = observation_dists
        self._initial_marginals = initial_marginals
        super(ForwardMessenger, self).__init__(f, phi=phi, theta=theta)

    def _condition(self, trace=None, observations=None):
        super(ForwardMessenger, self)._condition(trace, observations)
        if self._initial_marginals and (self._latent % 0 in self.trace):
            initial_note = self.trace.annotation(self._initial_marginals[0],
                                                 self._latent % 0)
            initial_note['forward_joint_marginals'] = self._initial_marginals[1]

    def forward(self, *args, **kwargs):
        results = super(ForwardMessenger, self).forward(*args, **kwargs)

        t = args[-1] + 1
        num_particles = self.trace[self._latent % t].value.shape[0]
        if t == 1:
            prev_note = self.trace.annotation(self._initial_marginals[0],
                                              self._latent % (t-1))
        else:
            prev_note = self.trace.annotation(self.name,
                                              self._latent % (t-1))
        latent_note = self.trace.annotation(self.name, self._latent % t)
        support = self.trace[self._latent % t].dist.enumerate_support()
        support = range(len(support))

        marginals = torch.zeros(num_particles, len(support), len(support))
        for prev, current in itertools.product(support, support):
            marginals[:, current, prev] =\
                self.transition(prev, current) +\
                prev_note['forward_joint_marginals'][:, prev]
        marginals = log_sum_exp(marginals, dim=-1)

        alpha = torch.zeros(num_particles, len(support))
        observed = self.trace[self._observation % t].value
        for current in support:
            alpha[:, current] =\
                self.observation_dists[current].log_prob(observed) +\
                marginals[:, current]
        latent_note['forward_joint_marginals'] = alpha
