#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
from probtorch.util import log_mean_exp
import torch

import combinators
from combinators import ParticleTrace
import importance
from importance import ResamplerTrace

class IndependentMH(combinators.Model):
    def __init__(self, model, proposal, num_iterations=1, trainable={},
                 hyper={}):
        super(IndependentMH, self).__init__(model, trainable, hyper)
        self._proposal = proposal
        self._num_iterations = num_iterations

    def forward(self, *args, **kwargs):
        elbos = torch.zeros(self._num_iterations)
        samples = list(range(self._num_iterations))
        original_trace = kwargs.get('trace', None)

        for i in range(self._num_iterations):
            self._proposal.simulate(
                trace=ResamplerTrace(ancestor=original_trace),
                reparameterized=False
            )
            kwargs['trace'] = ResamplerTrace(original_trace.num_particles,
                                             guide=self._proposal.trace,
                                             data=original_trace.data)
            sample = super(IndependentMH, self).forward(*args, **kwargs)

            elbo = self._function.marginal_log_likelihood()
            acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
            if torch.bernoulli(acceptance) == 1 or i == 0:
                elbos[i] = elbo
                samples[i] = sample
            else:
                elbos[i] = elbos[i-1]
                samples[i] = samples[i-1]

        result = []
        for i in range(self._num_iterations):
            particle = np.random.randint(0, self.trace.num_particles)
            result.append([v[particle] for v in samples[i]])
        return [torch.stack([result[i][j] for i in range(self._num_iterations)],
                            dim=0) for j, _ in enumerate(samples[0])], elbos
