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

class StepwiseImportanceResampler(importance.ImportanceResampler):
    def __init__(self, model, proposal=None, trainable={}, hyper={},
                 resample_factor=2):
        super(StepwiseImportanceResampler, self).__init__(model, proposal,
                                                          trainable, hyper,
                                                          resample_factor)

    @property
    def importance_observations(self):
        fresh = self.trace.fresh_variables
        return set(fresh.intersection(self.trace.observations))

    @property
    def importance_latents(self):
        fresh = self.trace.fresh_variables
        return set(fresh.intersection(self.trace.latents))

    def marginal_log_likelihood(self):
        log_weights = torch.stack(self.trace.saved_log_weights, dim=-1)
        return log_mean_exp(log_weights, dim=0).sum()

class SequentialMonteCarlo(combinators.Model):
    def __init__(self, step_model, T, step_proposal=None,
                 initializer=None, resample_factor=2):
        resampled_step = StepwiseImportanceResampler(
            step_model, step_proposal, resample_factor=resample_factor
        )
        step_sequence = combinators.Model.sequence(resampled_step, T)
        if initializer:
            model = combinators.Model.compose(step_sequence, initializer,
                                              intermediate_name='initializer')
        else:
            model = step_sequence
        super(SequentialMonteCarlo, self).__init__(model)
        self.resampled_step = resampled_step
        self.initializer = initializer

    def importance_weight(self, observations=None, latents=None):
        return self.resampled_step.importance_weight(observations, latents)

    def marginal_log_likelihood(self):
        return self.resampled_step.marginal_log_likelihood()

def variational_smc(num_particles, sampler, num_iterations, data,
                    use_cuda=True, lr=1e-6, inclusive_kl=False):
    optimizer = torch.optim.Adam(list(sampler.proposal.parameters()), lr=lr)

    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        sampler.simulate(trace=ResamplerTrace(num_particles, data=data),
                         proposal_guides=not inclusive_kl,
                         reparameterized=False)
        inference = sampler.trace
        if inclusive_kl:
            latents = sampler.proposal.latents()
            eubo = inference.log_joint(nodes=latents, reparameterized=False)
            joint_vars = latents + sampler.model.observations()
            eubo = eubo - sampler.model.trace.log_joint(
                nodes=joint_vars, reparameterized=False
            )
            eubo = -eubo.mean(dim=0)
            logging.info('Variational SMC EUBO=%.8e at epoch %d', eubo, t + 1)
            eubo.backward()
        else:
            elbo = sampler.model.marginal_log_likelihood()
            logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)
            (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    return inference, sampler.proposal.args_vardict()
