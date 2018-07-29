#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
from probtorch.util import log_mean_exp
import torch

import combinators
import importance
from importance import ResamplerTrace

class StepwiseImportanceResampler(importance.ImportanceResampler):
    def __init__(self, f, trainable={}, hyper={}):
        super(StepwiseImportanceResampler, self).__init__(f, trainable, hyper)

    def importance_weight(self):
        observations = self.observations()[-1:]
        latents = self.latents()[-1:]
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

    def marginal_log_likelihood(self):
        return list(self._function.children())[0].marginal_log_likelihood()

def variational_smc(num_particles, model_init, smc_sequence, num_iterations,
                    data, *args, use_cuda=True, lr=1e-6):
    optimizer = torch.optim.Adam(list(model_init.parameters()) +\
                                 list(smc_sequence.parameters()), lr=lr)

    model_init.train()
    smc_sequence.train()
    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_sequence.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        inference = ResamplerTrace(num_particles)

        vs = model_init(*args, trace=inference, guide=data)
        vs = smc_sequence(initializer=vs, trace=inference, guide=data)

        inference = smc_sequence.trace
        elbo = smc_sequence.marginal_log_likelihood()
        logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)

        (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_sequence.cpu()
    model_init.eval()
    smc_sequence.eval()

    return inference, model_init.args_vardict()

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

        for i in range(self._num_iterations):
            num_particles = kwargs['trace'].num_particles if 'trace' in kwargs\
                            else 1
            kwargs['trace'] = ResamplerTrace(num_particles)

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
