#!/usr/bin/env python3

import collections
import logging

from probtorch.stochastic import RandomVariable
import torch
import torch.distributions as dists
from torch.nn.functional import log_softmax

from ..sampler import Sampler
from . import inference
from ..model import foldable
from .. import utils

class Importance(inference.Inference):
    def __init__(self, target, proposal):
        super(Importance, self).__init__(target)
        assert isinstance(proposal, Sampler)
        assert proposal.batch_shape == target.batch_shape
        self.add_module('proposal', proposal)

    def forward(self, *args, **kwargs):
        _, xi, _ = self.proposal(*args, **kwargs)
        return self.target.cond(xi)(*args, **kwargs)

    def walk(self, f):
        return f(Importance(self.target.walk(f), self.proposal))

    def cond(self, qs):
        return Importance(self.target.cond(qs), self.proposal)

def importance(target, proposal):
    return Importance(target, proposal)

def collapsed_index_select(tensor, batch_shape, ancestors):
    tensor, unique = utils.batch_collapse(tensor, batch_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(batch_shape + unique)

def index_select_rv(rv, batch_shape, ancestors):
    result = rv
    if not rv.observed:
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

        particle_logs, _ = utils.batch_collapse(log_weights, self.batch_shape)
        particle_logs = log_softmax(particle_logs, dim=0)
        particle_dist = dists.Categorical(logits=particle_logs)
        ancestors = particle_dist.sample(particle_logs.shape)

        zs = list(zs)
        for i, z in enumerate(zs):
            zs[i] = collapsed_index_select(z, self.batch_shape, ancestors)
        if multiple_zs:
            zs = tuple(zs)
        else:
            zs = zs[0]

        resampler = lambda rv: index_select_rv(rv, self.batch_shape, ancestors)
        trace_resampler = lambda _, trace: utils.trace_map(trace, resampler)
        xi = xi.map(trace_resampler)

        log_weights = utils.batch_expand(utils.marginalize_all(log_weights),
                                         log_weights.shape)
        return zs, xi, log_weights

    def walk(self, f):
        return Resample(self.target.walk(f))

    def cond(self, qs):
        return Resample(self.target.cond(qs))

def resample(target):
    return Resample(target)

def resample_proposed(target, proposal):
    return resample(importance(target, proposal))

def smc(target):
    return target.walk(Resample)

def step_smc(sampler, initializer=None):
    resampler = Resample(sampler)
    return foldable.Step(resampler, initializer=initializer)

def reduce_smc(stepwise, step_generator, initializer=None):
    smc_foldable = step_smc(stepwise, initializer)
    return foldable.Reduce(smc_foldable, step_generator)

def variational_importance(sampler, num_iterations, data, use_cuda=True,
                           lr=1e-6, inclusive_kl=False, patience=50):
    optimizer = torch.optim.Adam(list(sampler.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-6, patience=patience, verbose=True,
        mode='min' if inclusive_kl else 'max',
    )

    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    bounds = list(range(num_iterations))
    for t in range(num_iterations):
        optimizer.zero_grad()

        _, xi, log_weight = sampler(data=data)

        bound = -utils.marginalize_all(log_weight)
        bound_name = 'EUBO' if inclusive_kl else 'ELBO'
        bounds[t] = bound if inclusive_kl else -bound
        logging.info('%s=%.8e at epoch %d', bound_name, bounds[t], t + 1)
        bound.backward()
        optimizer.step()
        scheduler.step(bounds[t])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    trained_params = sampler.args_vardict(False)

    return xi, trained_params, bounds
