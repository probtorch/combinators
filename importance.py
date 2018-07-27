#!/usr/bin/env python3

import collections

import probtorch
from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

def sampled_latent(rv, trace, guide):
    if isinstance(guide, probtorch.Trace):
        generative = rv in guide
    else:
        generative = True
    return not trace[rv].observed and generative

class ImportanceSampler(combinators.Model):
    def __init__(self, f, phi={}, theta={}):
        super(ImportanceSampler, self).__init__(f, phi, theta)
        self.log_weights = collections.OrderedDict()

    @property
    def _num_particles(self):
        if self.log_weights:
            return list(self.log_weights.items())[0][1].shape[0]
        return 1

    def observations(self):
        return [rv for rv in self.trace.variables() if self.trace[rv].observed\
                and self.trace.has_annotation(self.name, rv)]

    def latents(self):
        return [rv for rv in self.trace.variables()\
                if not self.trace[rv].observed and\
                self.trace.has_annotation(self.name, rv)]

    def importance_weight(self, observations=None, latents=None):
        if not observations:
            observations = self.observations()
        if not latents:
            latents = self.latents()
        log_likelihood = self.trace.log_joint(nodes=observations,
                                              reparameterized=False)
        log_proposal = self.trace.log_joint(nodes=latents,
                                            reparameterized=False)
        # If the guide is given by a generative model, use it, otherwise
        # perform likelihood weighting
        if isinstance(self.guide, probtorch.Trace):
            log_generative = utils.counterfactual_log_joint(self.guide,
                                                            self.trace, latents)
        else:
            log_generative = log_proposal
        log_generative = log_generative.to(log_proposal).mean(dim=0)

        self.log_weights[str(latents)] = log_likelihood + log_generative -\
                                         log_proposal
        return self.log_weights[str(latents)]

    def marginal_log_likelihood(self):
        return log_mean_exp(self.log_weights[str(self.latents())], dim=0)

    def forward(self, *args, **kwargs):
        results = super(ImportanceSampler, self).forward(*args, **kwargs)
        resampled_trace, resampled_weights = self.trace.resample(
            self.importance_weight()
        )
        self.log_weights[-1] = resampled_weights
        self.ancestor.condition(trace=resampled_trace)
        return results
