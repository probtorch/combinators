#!/usr/bin/env python3

import collections
import logging

import probtorch
from probtorch.util import log_mean_exp
import torch

import combinators
import importance
from importance import ResamplerTrace

class SequentialImportanceResampler(importance.ImportanceResampler):
    def __init__(self, f, phi={}, theta={}):
        super(SequentialImportanceResampler, self).__init__(f, phi, theta)

    def importance_weight(self):
        observations = self.observations()[-1:]
        latents = self.latents()[-1:]
        return super(SequentialImportanceResampler, self).importance_weight(
            observations, latents
        )

    def marginal_log_likelihood(self):
        latents = self.latents()
        log_weights = torch.zeros(len(latents), self._num_particles)
        for t, _ in enumerate(latents):
            log_weights[t] = self.log_weights[str(latents[t:t+1])]
        return log_mean_exp(log_weights, dim=0).sum()

def smc(step, T):
    resampled_step = SequentialImportanceResampler(step)
    return combinators.Model.sequence(resampled_step, T)

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
        elbo = list(smc_sequence.children())[0].marginal_log_likelihood()
        logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)

        (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_sequence.cpu()
    model_init.eval()
    smc_sequence.eval()

    return inference, model_init.args_vardict()

def particle_mh(num_particles, model_init, smc_sequence, num_iterations, data,
                *args, use_cuda=True):
    elbos = torch.zeros(num_iterations)
    samples = list(range(num_iterations))

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_sequence.cuda()

    for i in range(num_iterations):
        inference = ResamplerTrace(num_particles)

        vs = model_init(*args, trace=inference, guide=data)
        vs = smc_sequence(initializer=vs, trace=inference, guide=data)

        inference = smc_sequence.trace
        elbo = list(smc_sequence.children())[0].marginal_log_likelihood()

        acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
        if (torch.bernoulli(acceptance) == 1).sum() > 0 or i == 0:
            elbos[i] = elbo
            samples[i] = vs
        else:
            elbos[i] = elbos[i-1]
            samples[i] = samples[i-1]

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_sequence.cpu()

    return samples, elbos, inference
