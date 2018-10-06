#!/usr/bin/env python3

import collections
import logging

import probtorch
from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

class ImportanceTrace(combinators.ConditionedTrace):
    @property
    def weighting_variables(self):
        return self.variables()

    def log_proper_weight(self):
        nodes = list(self.weighting_variables)
        latents = [rv for rv in nodes if rv in self.latents]
        priors = [rv for rv in latents if self.guided(rv) is None]
        guided = [rv for rv in latents if self.guided(rv) is not None]

        generative_joint = self.log_joint(nodes=nodes, reparameterized=False)
        prior_joint = self.log_joint(nodes=priors, reparameterized=False)
        guide_joint = self.guide.log_joint(nodes=guided, reparameterized=False)\
                      if self.guide is not None else 0.0

        log_weight = generative_joint - (prior_joint + guide_joint)
        if not isinstance(log_weight, torch.Tensor):
            return torch.zeros(self.batch_shape).to(self.device)
        return log_weight

    def marginal_log_likelihood(self):
        return log_mean_exp(self.log_proper_weight())

class ImportanceSampler(combinators.Model):
    def __init__(self, model, proposal=None, trainable={}, hyper={}):
        super(ImportanceSampler, self).__init__(model, trainable, hyper)
        self._proposal = proposal

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
        return log_mean_exp(self.log_proper_weight(), dim=0)

class ResamplerTrace(ImportanceTrace):
    def __init__(self, num_particles=1, guide=None, data=None, ancestor=None,
                 log_weights=None):
        if ancestor is not None:
            num_particles = ancestor.num_particles
            guide = ancestor.guide
            data = ancestor.data
        super(ResamplerTrace, self).__init__(num_particles, guide=guide,
                                             data=data)
        self._ancestor = ancestor

        if log_weights is not None:
            normalized_weights = log_softmax(log_weights, dim=0)
            resampler = torch.distributions.Categorical(logits=normalized_weights)
            ancestor_indices = resampler.sample((self.num_particles,))
        else:
            ancestor_indices = None
        if isinstance(ancestor_indices, torch.Tensor):
            self._ancestor_indices = ancestor_indices
        else:
            self._ancestor_indices = torch.arange(self.num_particles,
                                                  dtype=torch.long)
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

    @property
    def ancestor(self):
        return self._ancestor

    @property
    def ancestor_indices(self):
        return self._ancestor_indices

    def is_inherited(self, variable):
        return self.ancestor and variable in self.ancestor

    @property
    def weighting_variables(self):
        return [node for node in super(ResamplerTrace, self).weighting_variables
                if not self.is_inherited(node)]

    def resample(self):
        log_weights = self.log_proper_weight()
        result = ResamplerTrace(self.num_particles, ancestor=self,
                                log_weights=log_weights)
        return result, log_weights.index_select(0, result.ancestor_indices)

    def effective_sample_size(self):
        return (self.log_proper_weight()*2).exp().sum(dim=0).pow(-1)

    def marginal_log_likelihood(self):
        recent_weight = log_mean_exp(self.log_proper_weight(), dim=0)
        if self.ancestor is not None:
            return recent_weight + self.ancestor.marginal_log_likelihood()
        return recent_weight

class ImportanceResampler(ImportanceSampler):
    def __init__(self, model, proposal=None, trainable={}, hyper={},
                 resample_factor=2):
        super(ImportanceResampler, self).__init__(model, proposal, trainable,
                                                  hyper)
        self._trace = ResamplerTrace()
        self._resample_factor = resample_factor

    def forward(self, *args, **kwargs):
        results = super(ImportanceResampler, self).forward(*args, **kwargs)
        ess = self.trace.effective_sample_size()
        if ess < self.trace.num_particles / self._resample_factor:
            resampled_trace, _ = self.trace.resample()
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

    @classmethod
    def smc(cls, step_model, T, step_proposal=None, initializer=None,
            resample_factor=2):
        resampled_step = cls(step_model, step_proposal,
                             resample_factor=resample_factor)
        step_sequence = combinators.Model.sequence(resampled_step, T)
        if initializer:
            return combinators.Model.compose(step_sequence, initializer,
                                             intermediate_name='initializer')
        else:
            return step_sequence

def variational_importance(num_particles, sampler, num_iterations, data,
                           use_cuda=True, lr=1e-6, inclusive_kl=False,
                           patience=50):
    optimizer = torch.optim.Adam(list(sampler.proposal.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-6, patience=patience, verbose=True,
        mode='min' if inclusive_kl else 'max',
    )

    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    bounds = list(range(num_iterations))
    for t in range(num_iterations):
        optimizer.zero_grad()

        sampler.simulate(trace=ResamplerTrace(num_particles, data=data),
                         proposal_guides=not inclusive_kl,
                         reparameterized=False)
        inference = sampler.trace

        bound = -inference.marginal_log_likelihood()
        bound_name = 'EUBO' if inclusive_kl else 'ELBO'
        signed_bound = bound if inclusive_kl else -bound
        logging.info('%s=%.8e at epoch %d', bound_name, signed_bound, t + 1)
        bound.backward()
        optimizer.step()
        bounds[t] = bound if inclusive_kl else -bound
        scheduler.step(bounds[t])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    trained_params = sampler.proposal.args_vardict(inference.batch_shape)

    return inference, trained_params, bounds
