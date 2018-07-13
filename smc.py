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
        super(ParticleTrace, self).__init__()
        self._num_particles = num_particles

    @property
    def num_particles(self):
        return self._num_particles

    def variable(self, Dist, *args, **kwargs):
        args = [arg.expand(self.num_particles, *arg.shape)
                if isinstance(arg, torch.Tensor) and
                (len(arg.shape) < 1 or arg.shape[0] != self.num_particles)
                else arg for arg in args]
        kwargs = {k: v.expand(self.num_particles, *v.shape)
                     if isinstance(v, torch.Tensor) and
                     (len(v.shape) < 1 or v.shape[0] != self.num_particles)
                     else v for k, v in kwargs.items()}
        return super(ParticleTrace, self).variable(Dist, *args, **kwargs)

    def log_joint(self, *args, **kwargs):
        return super(ParticleTrace, self).log_joint(*args, sample_dim=0,
                                                    **kwargs)

    def resample(self, log_weights):
        normalized_weights = log_softmax(log_weights, dim=0)
        resampler = torch.distributions.Categorical(logits=normalized_weights)
        ancestor_indices = resampler.sample((self.num_particles,))

        result = ParticleTrace(self.num_particles)
        result._modules = self._modules
        result._stack = self._stack
        for i, key in enumerate(self.variables()):
            rv = self[key] if key is not None else self[i]
            if not rv.observed:
                value = rv.value.index_select(0, ancestor_indices)
                sample = RandomVariable(rv.dist, value, rv.observed, rv.mask,
                                        rv.reparameterized)
            else:
                sample = rv
            if key is not None:
                result[key] = sample
            else:
                result[i] = sample

        return result, log_weights.index_select(0, ancestor_indices)

    def squeeze(self):
        result = combinators.GraphingTrace()
        result._modules = self._modules
        result._stack = self._stack

        for i, key in enumerate(self.variables()):
            if key is not None:
                rv = self[key]
                result[key] = RandomVariable(rv.dist, rv.value.median(dim=0)[0],
                                             rv.observed, rv.mask,
                                             rv.reparameterized)
            else:
                rv = self[i]
                result[i] = RandomVariable(rv.dist, rv.value.median(dim=0)[0],
                                           rv.observed, rv.mask,
                                           rv.reparameterized)

        return result

def smc(step, retrace):
    stepper = combinators.Model.compose(
        combinators.Model(retrace),
        importance.ImportanceSampler(step),
    )
    return combinators.Model.partial(combinators.Model(combinators.sequence),
                                     stepper)

def variational_smc(num_particles, model_init, smc_run, num_iterations, T,
                    params, data, *args, use_cuda=True, lr=1e-6):
    model_init = combinators.Model(model_init, params, {})
    optimizer = torch.optim.Adam(list(model_init.parameters()), lr=lr)

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_run.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_run.condition(trace=inference, observations=data)

        smc_run(T, *model_init(*args, T))
        inference = smc_run.trace
        elbo = list(smc_run.result.children())[0].marginal_log_likelihood()
        logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)

        (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_run.cpu()

    return inference, model_init.args_vardict()

def particle_mh(num_particles, model_init, smc_run, num_iterations, T, params,
                data, *args, use_cuda=True):
    model_init = combinators.Model(model_init, params, {})
    elbos = torch.zeros(num_iterations)
    samples = list(range(num_iterations))

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()
        smc_run.cuda()

    for i in range(num_iterations):
        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_run.condition(trace=inference, observations=data)

        vs = model_init(*args, T)

        vs = smc_run(T, *vs)
        inference = smc_run.trace
        elbo = list(smc_run.result.children())[0].marginal_log_likelihood()

        acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
        if (torch.bernoulli(acceptance) == 1).sum() > 0 or i == 0:
            elbos[i] = elbo
            samples[i] = vs
        else:
            elbos[i] = elbos[i-1]
            samples[i] = samples[i-1]

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        smc_run.cpu()

    return samples, elbos, inference
