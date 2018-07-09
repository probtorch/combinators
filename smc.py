#!/usr/bin/env python3

import logging

import probtorch
from probtorch.stochastic import RandomVariable, Trace
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

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
        resampler = torch.distributions.Categorical(logits=log_weights)
        ancestor_indices = resampler.sample((self.num_particles,))

        result = ParticleTrace(self.num_particles)
        result._modules = self._modules
        result._stack = self._stack
        for i, key in enumerate(self.variables()):
            rv = self[key]
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

        return result

def importance_weight(trace, conditions, t=-1):
    observations = [rv for rv in trace.variables() if trace[rv].observed]
    latents = [rv for rv in trace.variables() if not trace[rv].observed and
               rv in conditions]
    log_likelihood = trace.log_joint(nodes=[observations[t]])
    log_proposal = trace.log_joint(nodes=[latents[t]])
    log_generative = utils.counterfactual_log_joint(conditions, trace,
                                                    [latents[t]])
    log_generative = log_generative.to(log_proposal).mean(dim=0)

    return log_softmax(log_likelihood + log_generative - log_proposal, dim=0)

def smc(step, retrace):
    def resample(*args, **kwargs):
        this = kwargs['this']
        resampled_trace = this.trace.resample(importance_weight(
            this.trace, this.observations
        ))
        this.ancestor.condition(trace=resampled_trace)
        return args
    resample = combinators.Model(resample)
    stepper = combinators.Model.compose(
        combinators.Model.compose(retrace, resample),
        combinators.Model(step),
    )
    return combinators.Model.partial(combinators.Model(combinators.sequence),
                                     stepper)

def marginal_log_likelihood(trace, conditions, T):
    log_weights = torch.zeros(T, trace.num_particles)
    for t in range(T):
        log_weights[t] = importance_weight(trace, conditions, t)
    return log_mean_exp(log_weights, dim=1).sum()

def variational_smc(num_particles, model_init, smc_run, num_iterations, T,
                    params, data, *args):
    model_init = combinators.Model(model_init, params, {})
    optimizer = torch.optim.Adam(list(model_init.parameters()), lr=1e-6)

    if torch.cuda.is_available():
        model_init.cuda()
        smc_run.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_run.condition(trace=inference, observations=data)

        smc_run(T, *model_init(*args, T))
        inference = smc_run.trace
        elbo = marginal_log_likelihood(inference, data, T)
        logging.info('Variational SMC ELBO=%.8e at epoch %d', elbo, t + 1)

        (-elbo).backward()
        optimizer.step()

    if torch.cuda.is_available():
        model_init.cpu()
        smc_run.cpu()

    return inference, model_init.args_vardict()

def particle_mh(num_particles, model_init, smc_run, num_iterations, T, params,
                data, *args):
    model_init = combinators.Model(model_init, params, {})
    elbos = torch.zeros(num_iterations)
    samples = list(range(num_iterations))

    if torch.cuda.is_available():
        model_init.cuda()
        smc_run.cuda()

    for i in range(num_iterations):
        inference = ParticleTrace(num_particles)
        model_init.condition(trace=inference, observations=data)
        smc_run.condition(trace=inference, observations=data)

        vs = model_init(*args, T)

        vs = smc_run(T, *vs)
        inference = smc_run.trace
        elbo = marginal_log_likelihood(inference, data, T)

        acceptance = torch.min(torch.ones(1), torch.exp(elbo - elbos[i-1]))
        if (torch.bernoulli(acceptance) == 1).sum() > 0 or i == 0:
            elbos[i] = elbo
            samples[i] = vs
        else:
            elbos[i] = elbos[i-1]
            samples[i] = samples[i-1]

    if torch.cuda.is_available():
        model_init.cpu()
        smc_run.cpu()

    return samples, elbos, inference
