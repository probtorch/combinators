#!/usr/bin/env python3

import collections

import probtorch
from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

class ImportanceSampler(combinators.Model):
    def __init__(self, model, proposal=None, trainable={}, hyper={}):
        super(ImportanceSampler, self).__init__(model, trainable, hyper)
        self._proposal = proposal

    @property
    def importance_observations(self):
        return set(self.trace.observations)

    @property
    def importance_latents(self):
        return set(self.trace.latents)

    def forward(self, *args, **kwargs):
        kwargs['separate_traces'] = True
        if self.parent:
            kwargs['trace'] = self.trace

        if kwargs.pop('proposal_guides', True):
            if self.proposal:
                self._proposal(*args, **kwargs)

                inference = self._proposal.trace
                generative = combinators.ConditionedTrace.clamp(inference)

                kwargs = {**kwargs, 'trace': generative}
            result = self._function(*args, **kwargs)
            kwargs['trace'] = self.model.trace
        else:
            result = self._function(*args, **kwargs)
            if self.proposal:
                generative = self._function.trace
                inference = combinators.ConditionedTrace.clamp(generative)

                kwargs = {**kwargs, 'trace': inference}
                result = self._proposal(*args, **kwargs)
                kwargs['trace'] = self.proposal.trace
            else:
                kwargs['trace'] = self.model.trace
        if not self.parent:
            self._trace = kwargs['trace']
        return result

    @property
    def model(self):
        return self._function

    @property
    def proposal(self):
        return self._proposal

    def marginal_log_likelihood(self):
        return log_mean_exp(self.importance_weight(), dim=0)

class ResamplerTrace(combinators.ConditionedTrace):
    def __init__(self, num_particles=1, guide=None, data=None, ancestor=None,
                 log_weights=None):
        if ancestor is not None:
            num_particles = ancestor.num_particles
            guide = ancestor.guide
            data = ancestor.data
        super(ResamplerTrace, self).__init__(num_particles, guide=guide,
                                             data=data)
        self._ancestor = ancestor
        self._saved_log_weights = []

        if log_weights is not None:
            self._log_weights = log_weights
            normalized_weights = log_softmax(log_weights, dim=0)
            resampler = torch.distributions.Categorical(logits=normalized_weights)
            ancestor_indices = resampler.sample((self.num_particles,))
        else:
            self._log_weights = None
            ancestor_indices = None
        if isinstance(ancestor_indices, torch.Tensor):
            self._ancestor_indices = ancestor_indices
        else:
            self._ancestor_indices = torch.arange(self._num_particles,
                                                  dtype=torch.long)
        self._fresh_variables = set()
        if ancestor:
            self._saved_log_weights = ancestor.saved_log_weights
            self._modules = ancestor._modules
            self._stack = ancestor._stack
            for i, key in enumerate(ancestor.variables()):
                rv = ancestor[key] if key is not None else ancestor[i]
                if not ancestor.observed(rv):
                    value = rv.value.index_select(0, self.ancestor_indices)
                    sample = RandomVariable(rv.dist, value, rv.observed,
                                            rv.mask, rv.reparameterized)
                else:
                    sample = rv
                if key is not None:
                    self[key] = sample
                else:
                    self[i] = sample

    def variable(self, Dist, *args, **kwargs):
        if 'name' in kwargs:
            self._fresh_variables.add(kwargs['name'])
        return super(ResamplerTrace, self).variable(Dist, *args, **kwargs)

    @property
    def ancestor(self):
        return self._ancestor

    @property
    def fresh_variables(self):
        return self._fresh_variables

    @property
    def ancestor_indices(self):
        return self._ancestor_indices

    @property
    def saved_log_weights(self):
        return self._saved_log_weights

    def save_importance_weight(self, observations=None, latents=None):
        log_weights = self.importance_weight(observations, latents)
        self._saved_log_weights.append(log_weights)
        self._fresh_variables = set()
        return log_weights

    def resample(self, observations=None, latents=None):
        log_weights = self.importance_weight(observations, latents)
        result = ResamplerTrace(self.num_particles, ancestor=self,
                                log_weights=log_weights)
        return result, log_weights.index_select(0, result.ancestor_indices)

    def effective_sample_size(self, observations=None, latents=None,
                              save=False):
        if save:
            log_weights = self.save_importance_weight(observations, latents)
        else:
            log_weights = self.importance_weight(observations, latents)
        return (log_weights*2).exp().sum(dim=0).pow(-1)

class ImportanceResampler(ImportanceSampler):
    def __init__(self, model, proposal=None, trainable={}, hyper={},
                 resample_factor=2):
        super(ImportanceResampler, self).__init__(model, proposal, trainable,
                                                  hyper)
        self._trace = ResamplerTrace()
        self._resample_factor = resample_factor

    def forward(self, *args, **kwargs):
        results = super(ImportanceResampler, self).forward(*args, **kwargs)
        observations = self.importance_observations
        latents = self.importance_latents
        ess = self.trace.effective_sample_size(observations, latents, True)
        if ess < self.trace.num_particles / self._resample_factor:
            resampled_trace, _ = self.trace.resample(observations, latents)
            self.ancestor._condition_all(trace=resampled_trace)

            results = list(results)
            for i, var in enumerate(results):
                if isinstance(var, torch.Tensor):
                    ancestor_indices = self.trace.ancestor_indices.to(
                        device=var.device
                    )
                    results[i] = var.index_select(0, ancestor_indices)
            results = tuple(results)
        return results
