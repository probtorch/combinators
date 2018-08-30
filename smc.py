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

    def importance_weight(self, observations=None, latents=None):
        fresh = self.trace.fresh_variables
        if not observations:
            observations = list(fresh.intersection(self.observations()))
        if not latents:
            latents = list(fresh.intersection(self.latents()))
        return super(StepwiseImportanceResampler, self).importance_weight(
            observations, latents
        )

    def marginal_log_likelihood(self):
        log_weights = torch.zeros(len(self.log_weights), self._num_particles)
        for t, latent in enumerate(self.log_weights):
            log_weights[t] = self.log_weights[latent]
        return log_mean_exp(log_weights, dim=0).sum()

class SequentialMonteCarlo(combinators.Model):
    def __init__(self, step, T, initializer=None, resample_factor=2):
        resampled_step = StepwiseImportanceResampler(
            step.model, step.proposal, resample_factor=resample_factor
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

def variational_smc(num_particles, model, proposal, num_iterations, data,
                    use_cuda=True, lr=1e-6, inclusive_kl=False):
    optimizer = torch.optim.Adam(list(proposal.parameters()), lr=lr)
    parameters = proposal.args_vardict(False).keys()

    model.train()
    proposal.train()
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
        proposal.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        if inclusive_kl:
            model.simulate(trace=ResamplerTrace(num_particles, data=data),
                           reparameterized=False)
            inference = ResamplerTrace(num_particles, guide=model.trace,
                                       data=data)
            proposal(trace=inference)
            inference = proposal.trace

            latents = proposal.latents()
            eubo = inference.log_joint(nodes=latents, reparameterized=False)
            joint_vars = latents + model.observations()
            eubo = eubo - model.trace.log_joint(nodes=joint_vars,
                                                reparameterized=False)
            eubo = -eubo.mean(dim=0)
            logging.info('Variational SMC EUBO=%.8e at epoch %d', eubo, t + 1)
            eubo.backward()
        else:
            proposal.simulate(trace=ResamplerTrace(num_particles, data=data),
                              reparameterized=False)
            inference = ResamplerTrace(num_particles, guide=proposal.trace,
                                       data=data)
            model(trace=inference)
            inference = model.trace

            elbo = model.marginal_log_likelihood()
            logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)
            (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model.cpu()
        proposal.cpu()
    model.eval()
    proposal.eval()

    return inference, proposal.args_vardict()
