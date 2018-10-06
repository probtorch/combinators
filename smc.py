#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
from probtorch.util import log_mean_exp
import torch

import combinators
from combinators import BroadcastingTrace
import importance
from importance import ResamplerTrace

class SequentialMonteCarlo(combinators.Model):
    def __init__(self, step_model, T, step_proposal=None,
                 initializer=None, resample_factor=2):
        resampled_step = importance.ImportanceResampler(
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

def variational_smc(num_particles, sampler, num_iterations, data,
                    use_cuda=True, lr=1e-6, inclusive_kl=False):
    optimizer = torch.optim.Adam(list(sampler.proposal.parameters()), lr=lr)
    parameters = sampler.proposal.args_vardict(False).keys()

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
            latents = [latent for latent in inference.latents
                       if any([latent in param for param in parameters])]
            eubo = inference.importance_weight(None, latents=latents)
            eubo = -eubo.mean(dim=0)
            logging.info('Variational SMC EUBO=%.8e at epoch %d', eubo, t + 1)
            eubo.backward()
        else:
            elbo = inference.marginal_log_likelihood()
            logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)
            (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    return inference, sampler.proposal.args_vardict(inference.batch_shape)
