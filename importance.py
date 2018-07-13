#!/usr/bin/env python3

import collections

from probtorch.util import log_mean_exp
import torch

import combinators
import utils

class ImportanceSampler(combinators.Model):
    def __init__(self, f, phi={}, theta={}):
        super(ImportanceSampler, self).__init__(f, phi, theta)
        self.log_weights = collections.OrderedDict()

    @property
    def _num_particles(self):
        if self.log_weights:
            return list(self.log_weights.items())[0][1].shape[0]
        return 1

    def importance_weight(self, t=-1):
        observation = [rv for rv in self.trace.variables()
                       if self.trace[rv].observed][t]
        latent = [rv for rv in self.trace.variables()
                  if not self.trace[rv].observed and rv in self.observations][t]
        log_likelihood = self.trace.log_joint(nodes=[observation],
                                              reparameterized=False)
        log_proposal = self.trace.log_joint(nodes=[latent],
                                            reparameterized=False)
        log_generative = utils.counterfactual_log_joint(self.observations,
                                                        self.trace, [latent])
        log_generative = log_generative.to(log_proposal).mean(dim=0)

        self.log_weights[latent] = log_likelihood + log_generative -\
                                   log_proposal
        return self.log_weights[latent]

    def marginal_log_likelihood(self):
        log_weights = torch.zeros(len(self.log_weights),
                                  self._num_particles)
        for t, lw in enumerate(self.log_weights.values()):
            log_weights[t] = lw
        return log_mean_exp(log_weights, dim=1).sum()

    def forward(self, *args, **kwargs):
        results = super(ImportanceSampler, self).forward(*args, **kwargs)
        resampled_trace, resampled_weights = self.trace.resample(
            self.importance_weight()
        )
        self.log_weights[-1] = resampled_weights
        self.ancestor.condition(trace=resampled_trace)
        return results
