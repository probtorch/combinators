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
        self.log_weights = collections.OrderedDict()

    def forward(self, *args, **kwargs):
        kwargs['separate_traces'] = True
        if self.parent:
            kwargs['trace'] = self.trace

        if kwargs.pop('proposal_guides', True):
            if self.proposal:
                self._proposal(*args, **kwargs)

                inference = self._proposal.trace
                generative = combinators.GuidedTrace.clamp(inference)

                kwargs = {**kwargs, 'trace': generative}
            result = self._function(*args, **kwargs)
        else:
            result = self._function(*args, **kwargs)
            if self.proposal:
                generative = self._function.trace
                inference = combinators.GuidedTrace.clamp(generative)

                kwargs = {**kwargs, 'trace': inference}
                result = self._proposal(*args, **kwargs)
        if not self.parent:
            self._trace = kwargs['trace']
        return result

    @property
    def _num_particles(self):
        if self.log_weights:
            return list(self.log_weights.items())[0][1].shape[0]
        return 1

    @property
    def model(self):
        return self._function

    @property
    def proposal(self):
        return self._proposal

    def importance_weight(self, observations=None, latents=None):
        if not observations:
            observations = self.observations()
        if not latents:
            latents = self.latents()
        log_likelihood = self.trace.log_joint(nodes=observations,
                                              reparameterized=False)
        log_prior = self.trace.log_joint(nodes=latents, normalize_guide=True,
                                         reparameterized=False)

        self.log_weights[str(latents)] = log_likelihood + log_prior
        return self.log_weights[str(latents)]

    def effective_sample_size(self, observations=None, latents=None):
        log_weights = self.importance_weight(observations, latents)
        return (log_weights.exp() ** 2).sum(dim=0).pow(-1)

    def marginal_log_likelihood(self):
        return log_mean_exp(self.log_weights[str(self.latents())], dim=0)

class ResamplerTrace(combinators.GuidedTrace):
    def __init__(self, num_particles=1, guide=None, data=None,
                 ancestor_indices=None, ancestor=None):
        if ancestor is not None:
            num_particles = ancestor.num_particles
            guide = ancestor.guide
            data = ancestor.data
        super(ResamplerTrace, self).__init__(num_particles, guide=guide,
                                             data=data)
        self._ancestor = ancestor
        if isinstance(ancestor_indices, torch.Tensor):
            self._ancestor_indices = ancestor_indices
        else:
            self._ancestor_indices = torch.arange(self._num_particles,
                                                  dtype=torch.long)
        self._fresh_variables = set()
        if ancestor:
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

    def resample(self, log_weights):
        normalized_weights = log_softmax(log_weights, dim=0)
        resampler = torch.distributions.Categorical(logits=normalized_weights)

        result = ResamplerTrace(
            self.num_particles,
            ancestor_indices=resampler.sample((self.num_particles,)),
            ancestor=self
        )
        return result, log_weights.index_select(0, result.ancestor_indices)

    def collapse(self):
        if self.ancestor:
            indices = self.ancestor.collapse()
            return indices.index_select(0, self.ancestor_indices)
        return self.ancestor_indices

class ImportanceResampler(ImportanceSampler):
    def __init__(self, model, proposal=None, trainable={}, hyper={},
                 resample_factor=2):
        super(ImportanceResampler, self).__init__(model, proposal, trainable,
                                                  hyper)
        self._trace = ResamplerTrace()
        self._resample_factor = resample_factor

    def forward(self, *args, **kwargs):
        results = super(ImportanceResampler, self).forward(*args, **kwargs)
        ess = self.effective_sample_size()
        if ess < self.trace.num_particles / self._resample_factor:
            resampled_trace, _ = self.trace.resample(self.importance_weight())
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
