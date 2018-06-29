#!/usr/bin/env python3

import probtorch
from probtorch.stochastic import RandomVariable, Trace
from probtorch.util import log_mean_exp
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

class ParticleTrace(probtorch.stochastic.Trace):
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
    latents = [rv for rv in trace.variables() if not trace[rv].observed]
    log_likelihood = trace.log_joint(nodes=observations[t:])
    log_proposal = trace.log_joint(nodes=latents[t:])
    log_generative = conditions.log_joint(nodes=latents[t:])

    return log_softmax(log_likelihood + log_generative - log_proposal, dim=0)

def smc(step, retrace):
    def resample(*args, **kwargs):
        trace = kwargs['trace']
        conditions = kwargs['conditions']
        trace = trace.resample(importance_weight(trace, conditions))
        return args + (trace,)
    resample = combinators.Inference(resample)
    return combinators.Inference.compose(
        combinators.Inference.compose(retrace, resample),
        step,
    )

def smc_run(smc_step, trace, conditions, T, *args):
    args = args
    for t in range(T):
        results = smc_step(*args, t+1, trace=trace, conditions=conditions)
        args = results[:-1]
        trace = results[-1]
    return trace

def marginal_log_likelihood(trace, conditions, T):
    log_weights = torch.zeros(T, trace.num_particles)
    for t in range(T):
        log_weights[t] = importance_weight(trace, conditions, t)
    return log_mean_exp(log_weights, dim=1).sum()

def variational_smc(num_particles, model_init, smc_step, num_iterations, T,
                    params, data, *args):
    model_init = combinators.Model(model_init, 'params', params, {})
    optimizer = torch.optim.Adam(list(model_init.parameters()), lr=1e-2)

    for _ in range(num_iterations):
        optimizer.zero_grad()

        inference = ParticleTrace(num_particles)
        vs = model_init(*args, T, trace=inference)

        inference = smc_run(smc_step, inference, data, T, *vs)
        elbo = marginal_log_likelihood(inference, data, T)

        (-elbo).backward()
        optimizer.step()

    return inference, model_init.args_vardict()

def particle_mh(num_particles, model_init, smc_step, num_iterations, T, params,
                data, *args):
    model_init = combinators.Model(model_init, 'params', params, {})
    elbos = torch.zeros(num_iterations)
    samples = arange(num_iterations)

    for i in range(num_iterations):
        inference = ParticleTrace(num_particles)
        vs = model_init(*args, T, trace=inference)

        inference = smc_run(smc_step, inference, data, T, *vs)
        elbo = marginal_log_likelihood(inference, data, T)

        acceptance = torch.max(torch.ones(1), torch.exp(elbo - elbos[i-1]))
        if (torch.bernoulli(acceptance) == 1).sum() > 0:
            elbos[i] = elbo
            samples[i] = vs
        else:
            elbos[i] = elbos[i-1]
            samples[i] = samples[i-1]

    return samples, elbos, inference
