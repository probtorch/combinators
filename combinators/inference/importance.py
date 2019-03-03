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
    log_omega_q = torch.zeros(*batch_shape)
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

def elbo(log_weight):
    return utils.batch_mean(log_weight)

def eubo(log_weight):
    ancestors, _ = utils.gumbel_max_resample(log_weight)
    eubo_particles = collapsed_index_select(log_weight, log_weight.shape,
                                            ancestors)
    return -utils.batch_mean(eubo_particles)

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

        bounds[t] = eubo(log_weight) if inclusive_kl else elbo(log_weight)
        bound_name = 'EUBO' if inclusive_kl else 'ELBO'
        logging.info('%s=%.8e at epoch %d', bound_name, bounds[t], t + 1)
        energy = bounds[t] if inclusive_kl else -bounds[t]
        energy.backward()
        optimizer.step()
        scheduler.step(bounds[t])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    trained_params = sampler.args_vardict(False)

    return xi, trained_params, bounds
