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
    def __init__(self, f, trainable={}, hyper={}):
        super(StepwiseImportanceResampler, self).__init__(f, trainable, hyper)

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
    def __init__(self, step, T):
        resampled_step = StepwiseImportanceResampler(step)
        super(SequentialMonteCarlo, self).__init__(
            combinators.Model.sequence(resampled_step, T)
        )
        self.resampled_step = resampled_step

    def importance_weight(self, observations=None, latents=None):
        return self.resampled_step.importance_weight(observations, latents)

    def marginal_log_likelihood(self):
        return self.resampled_step.marginal_log_likelihood()

def variational_smc(num_particles, model_init, smc_sequence, num_iterations,
                    data, proposal, use_cuda=True, lr=1e-6,
                    inclusive_kl=False):
    optimizer = torch.optim.Adam(list(model_init.parameters()) +\
                                 list(smc_sequence.parameters()) +\
                                 list(proposal.parameters()), lr=lr)

    model_init.train()
    smc_sequence.train()
    proposal.train()
    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_sequence.cuda()
        proposal.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        proposal.simulate(trace=ParticleTrace(num_particles),
                          reparameterized=False)
        inference = ResamplerTrace(num_particles, guide=proposal.trace,
                                   data=data)

        vs = model_init(trace=inference)
        vs = smc_sequence(initializer=vs, trace=inference)

        inference = smc_sequence.trace
        if inclusive_kl:
            latents = proposal.latents()
            hp_logq = inference.log_joint(nodes=latents, normalize_guide=True,
                                          reparameterized=False)
            hp_logq = -hp_logq.sum(dim=0)
            logging.info('Variational SMC H_p[log q]=%.8e at epoch %d', hp_logq,
                         t + 1)
            hp_logq.backward()
        else:
            elbo = smc_sequence.marginal_log_likelihood()
            logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)
            (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_sequence.cpu()
        proposal.cpu()
    model_init.eval()
    smc_sequence.eval()
    proposal.eval()

    return inference, proposal.args_vardict()

class ParticleMH(combinators.Model):
    def __init__(self, model_init, smc_sequence, num_iterations=1, trainable={},
                 hyper={}):
        smc_model = combinators.Model.compose(smc_sequence, model_init,
                                              intermediate_name='initializer')
        super(ParticleMH, self).__init__(smc_model, trainable, hyper)
        self._num_iterations = num_iterations
        self.add_module('_function', smc_model)

    def forward(self, *args, **kwargs):
        elbos = torch.zeros(self._num_iterations)
        samples = list(range(self._num_iterations))
        original_trace = kwargs.get('trace', None)

        for i in range(self._num_iterations):
            kwargs['trace'] = ResamplerTrace(ancestor=original_trace)

            vs = super(ParticleMH, self).forward(*args, **kwargs)

            elbo = self._function.wrapper.marginal_log_likelihood()
            acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
            if torch.bernoulli(acceptance) == 1 or i == 0:
                elbos[i] = elbo
                samples[i] = vs
            else:
                elbos[i] = elbos[i-1]
                samples[i] = samples[i-1]

        result = []
        for i in range(self._num_iterations):
            particle = np.random.randint(0, self.trace.num_particles)
            result.append([v[particle] for v in samples[i]])
        return [torch.stack([result[i][j] for i in range(self._num_iterations)],
                            dim=0) for j, _ in enumerate(samples[0])], elbos
