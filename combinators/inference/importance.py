#!/usr/bin/env python3

import collections
import logging

from probtorch.stochastic import RandomVariable
import torch
import torch.distributions as dists
from torch.nn.functional import log_softmax, softmax

from ..sampler import Sampler
from . import inference
from ..model import foldable
from .. import utils

def conditioning_factor(dest, src, batch_shape):
    sample_dims = tuple(range(len(batch_shape)))
    log_omega_q = torch.zeros(*batch_shape, device=src.device)
    for name in src:
        conditioned = [k for k in src[name].conditioned()]
        log_omega_q += src[name].log_joint(sample_dims=sample_dims,
                                           nodes=conditioned,
                                           reparameterized=False)
        if name in dest:
            reused = [k for k in dest[name] if k in src[name]]
            log_omega_q += src[name].log_joint(sample_dims=sample_dims,
                                               nodes=reused,
                                               reparameterized=False)
    return log_omega_q

class Propose(inference.Inference):
    def __init__(self, target, proposal):
        super(Propose, self).__init__(target)
        assert isinstance(proposal, Sampler)
        assert proposal.batch_shape == target.batch_shape
        self.add_module('proposal', proposal)

    def forward(self, *args, **kwargs):
        _, xiq, log_wq = self.proposal(*args, **kwargs)
        zs, xi, log_w = self.target.cond(xiq)(*args, **kwargs)
        log_omega_q = conditioning_factor(xi, xiq, self.target.batch_shape)
        return zs, xi, log_wq - log_omega_q + log_w

    def walk(self, f):
        return f(Propose(self.target.walk(f), self.proposal))

    def cond(self, qs):
        return Propose(self.target, self.proposal.cond(qs))

def propose(target, proposal):
    return Propose(target, proposal)

def collapsed_index_select(tensor, batch_shape, ancestors):
    tensor, unique = utils.batch_collapse(tensor, batch_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(batch_shape + unique)

def index_select_rv(rv, batch_shape, ancestors):
    result = rv
    if isinstance(rv, RandomVariable) and not rv.observed:
        value = collapsed_index_select(rv.value, batch_shape, ancestors)
        result = RandomVariable(rv.dist, value, rv.provenance, rv.mask,
                                rv.reparameterized)
    return result

class Resample(inference.Inference):
    def forward(self, *args, **kwargs):
        zs, xi, log_weights = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        ancestors, log_weights = utils.gumbel_max_resample(log_weights)

        zs = list(zs)
        for i, z in enumerate(zs):
            if isinstance(z, torch.Tensor):
                zs[i] = collapsed_index_select(z, self.batch_shape, ancestors)
        if multiple_zs:
            zs = tuple(zs)
        else:
            zs = zs[0]

        resampler = lambda rv: index_select_rv(rv, self.batch_shape, ancestors)
        trace_resampler = lambda _, trace: utils.trace_map(trace, resampler)
        xi = xi.map(trace_resampler)

        return zs, xi, log_weights

    def walk(self, f):
        return f(Resample(self.target.walk(f)))

    def cond(self, qs):
        return Resample(self.target.cond(qs))

def resample(target):
    return Resample(target)

def resample_proposed(target, proposal):
    return resample(propose(target, proposal))

def smc(target):
    selector = lambda m: isinstance(m, Propose)
    return target.apply(resample, selector)

def step_smc(target):
    selector = lambda m: isinstance(m, foldable.Step)
    return target.apply(resample, selector)

def dreg(log_weight, log_mean_estimator=False, alpha=torch.zeros(())):
    probs = utils.normalize_weights(log_weight).detach().exp()
    particles = (alpha * probs + (1 - 2 * alpha) * probs**2) * log_weight
    if log_mean_estimator:
        return utils.log_sum_exp(particles)
    return utils.batch_sum(particles)

def elbo(log_weight, log_mean_estimator=False, xi=None):
    if xi and xi.reparameterized():
        return dreg(log_weight, log_mean_estimator, alpha=torch.zeros(()))
    else:
        if log_mean_estimator:
            return utils.batch_marginalize(log_weight)
        return utils.batch_mean(log_weight)

def eubo(log_weight, log_mean_estimator=False, xi=None):
    if xi and xi.reparameterized():
        return -dreg(log_weight, log_mean_estimator, alpha=torch.ones(()))
    else:
        probs = utils.normalize_weights(log_weight).detach().exp()
        eubo_particles = probs * log_weight
        if log_mean_estimator:
            return utils.log_sum_exp(eubo_particles)
        return utils.batch_sum(eubo_particles)

def variational_importance(sampler, num_iterations, data, use_cuda=True, lr=1e-6,
                           bound='elbo', log_all_bounds=False, patience=50,
                           log_estimator=False):
    optimizer = torch.optim.Adam(list(sampler.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-6, patience=patience, verbose=True,
        mode='min' if bound == 'eubo' else 'max',
    )

    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    bounds = list(range(num_iterations))
    for t in range(num_iterations):
        optimizer.zero_grad()

        _, xi, log_weight = sampler(data=data)

        eubo_t = eubo(log_weight, log_estimator, xi=xi)
        elbo_t = elbo(log_weight, log_estimator, xi=xi)
        bounds[t] = {'eubo': eubo_t, 'elbo': elbo_t}
        if log_all_bounds:
            logging.info('ELBO=%.8e at epoch %d', bounds[t]['elbo'], t + 1)
            logging.info('EUBO=%.8e at epoch %d', bounds[t]['eubo'], t + 1)
        else:
            logging.info('%s=%.8e at epoch %d', bound.upper(), bounds[t][bound], t + 1)

        free_energy = bounds[t][bound]
        if bound == 'elbo':
            free_energy = -free_energy
        free_energy.backward()
        optimizer.step()
        scheduler.step(bounds[t][bound])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
        torch.cuda.empty_cache()
    sampler.eval()

    trained_params = sampler.args_vardict(False)
    bounds = (torch.stack([bs['elbo'] for bs in bounds], dim=0),
              torch.stack([bs['eubo'] for bs in bounds], dim=0))

    return xi, trained_params, bounds
