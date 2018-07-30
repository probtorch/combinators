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

class ResamplerTrace(combinators.ParticleTrace):
    def __init__(self, num_particles=1, ancestor_indices=None, ancestor=None):
        super(ResamplerTrace, self).__init__(num_particles=num_particles)
        if isinstance(ancestor_indices, torch.Tensor):
            self._ancestor_indices = ancestor_indices
        else:
            self._ancestor_indices = torch.arange(self._num_particles,
                                                  dtype=torch.long)
        if ancestor:
            self._modules = ancestor._modules
            self._stack = ancestor._stack
            for i, key in enumerate(ancestor.variables()):
                rv = ancestor[key] if key is not None else ancestor[i]
                if not rv.observed:
                    value = rv.value.index_select(0, self.ancestor_indices)
                    sample = RandomVariable(rv.dist, value, rv.observed,
                                            rv.mask, rv.reparameterized)
                else:
                    sample = rv
                if key is not None:
                    self[key] = sample
                else:
                    self[i] = sample

    @property
    def ancestor_indices(self):
        return self._ancestor_indices

    def resample(self, log_weights):
        normalized_weights = log_softmax(log_weights, dim=0)
        resampler = torch.distributions.Categorical(logits=normalized_weights)

        result = ResamplerTrace(self.num_particles,
                                resampler.sample((self.num_particles,)),
                                ancestor=self)
        return result, log_weights.index_select(0, result.ancestor_indices)

class ImportanceResampler(ImportanceSampler):
    def __init__(self, f, phi={}, theta={}):
        super(ImportanceResampler, self).__init__(f, phi, theta)
        self._trace = ResamplerTrace()

    def forward(self, *args, **kwargs):
        results = super(ImportanceResampler, self).forward(*args, **kwargs)
        resampled_trace, _ = self.trace.resample(
            self.importance_weight()
        )
        self.ancestor._condition_all(trace=resampled_trace, guide=self.guide)

        results = list(results)
        for i, var in enumerate(results):
            if isinstance(var, torch.Tensor):
                ancestor_indices = self.trace.ancestor_indices.to(
                    device=var.device
                )
                results[i] = var.index_select(0, ancestor_indices)
        return tuple(results)
