#!/usr/bin/env python3

import collections
import logging

import probtorch
from probtorch.stochastic import RandomVariable, Trace
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import importance

class ParticleTrace(combinators.GraphingTrace):
    def __init__(self, num_particles=1):
        super(ParticleTrace, self).__init__(num_particles=num_particles)
        self.ancestor_indices = torch.arange(self._num_particles,
                                             dtype=torch.long)

    def resample(self, log_weights):
        normalized_weights = log_softmax(log_weights, dim=0)
        resampler = torch.distributions.Categorical(logits=normalized_weights)
        self.ancestor_indices = resampler.sample((self.num_particles,))

        result = ParticleTrace(self.num_particles)
        result._modules = self._modules
        result._stack = self._stack
        for i, key in enumerate(self.variables()):
            rv = self[key] if key is not None else self[i]
            if not rv.observed:
                value = rv.value.index_select(0, self.ancestor_indices)
                sample = RandomVariable(rv.dist, value, rv.observed, rv.mask,
                                        rv.reparameterized)
            else:
                sample = rv
            if key is not None:
                result[key] = sample
            else:
                result[i] = sample

        return result, log_weights.index_select(0, self.ancestor_indices)

def smc(step, retrace):
    return combinators.Model.compose(
        combinators.Model(retrace),
        importance.ImportanceSampler(step),
    )

def variational_smc(num_particles, model_init, smc_step, num_iterations, T,
                    data, *args, use_cuda=True, lr=1e-6):
    optimizer = torch.optim.Adam(list(model_init.parameters()), lr=lr)

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_step.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_step.condition(trace=inference, observations=data)

        vs = model_init(*args, T)
        sequencer = combinators.Model.sequence(smc_step, T, *vs)
        sequencer.condition(trace=inference, observations=data)
        vs = sequencer()

        inference = smc_step.trace
        elbo = list(smc_step.children())[0].marginal_log_likelihood()
        logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)

        (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_step.cpu()

    return inference, model_init.args_vardict()

def particle_mh(num_particles, model_init, smc_step, num_iterations, T, params,
                data, *args, use_cuda=True):
    model_init = combinators.Model(model_init, params, {})
    elbos = torch.zeros(num_iterations)
    samples = list(range(num_iterations))

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_step.cuda()

    for i in range(num_iterations):
        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_step.condition(trace=inference, observations=data)

        vs = model_init(*args, T)

        sequencer = combinators.Model.sequence(smc_step, T, *vs)
        sequencer.condition(trace=inference, observations=data)
        vs = sequencer()

        inference = smc_step.trace
        elbo = list(smc_step.children())[0].marginal_log_likelihood()

        acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
        if (torch.bernoulli(acceptance) == 1).sum() > 0 or i == 0:
            elbos[i] = elbo
            samples[i] = vs
        else:
            elbos[i] = elbos[i-1]
            samples[i] = samples[i-1]

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_step.cpu()

    return samples, elbos, inference
