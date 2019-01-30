#!/usr/bin/env python3

import collections
import logging

from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp
import torch
import torch.distributions as dists
from torch.nn.functional import log_softmax

import combinators
import foldable
import trace_tries

def index_select_rv(rv, dim, indices):
    result = rv
    if not rv.observed:
        value = rv.value.index_select(dim, indices)
        result = RandomVariable(rv.dist, value, rv.observed, rv.mask,
                                rv.reparameterized)
    return result

class ImportanceResampler(combinators.InferenceSampler):
    def __init__(self, sampler, particle_shape):
        super(ImportanceResampler, self).__init__(sampler)
        self._particle_shape = particle_shape

    @property
    def particle_shape(self):
        return self._particle_shape

    def sample_prehook(self, trace, *args, **kwargs):
        return trace, args, kwargs

    def sample_hook(self, results, trace):
        weights = trace.normalized_log_weight()
        resampler = dists.Categorical(logits=weights)
        ancestor_indices = resampler.sample(self.particle_shape)
        results = [val.index_select(0, ancestor_indices) for val in results]
        trace_resampler = lambda k, rv: index_select_rv(rv, 0, ancestor_indices)
        return tuple(results), trace.map(trace_resampler)

def importance_with_proposal(proposal, model, particle_shape):
    scored_sampler = combinators.score_under_proposal(proposal, model)
    return ImportanceResampler(scored_sampler, particle_shape)

def smc(sampler, particle_shape, initializer=None):
    resampler = ImportanceResampler(sampler, particle_shape)
    return foldable.Foldable(resampler, initializer=initializer)

def reduce_smc(stepwise, particle_shape, step_generator, initializer=None):
    smc_foldable = smc(stepwise, particle_shape, initializer)
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

        trace = trace_tries.HierarchicalTrace()
        _, inference, _ = sampler.simulate(data=data, trace=trace)

        bound = -inference.marginal_log_likelihood()
        bound_name = 'EUBO' if inclusive_kl else 'ELBO'
        bounds[t] = bound if inclusive_kl else -bound
        logging.info('%s=%.8e at epoch %d', bound_name, bounds[t], t + 1)
        bound.backward()
        optimizer.step()
        scheduler.step(bounds[t])

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
    sampler.eval()

    trained_params = sampler.args_vardict()

    return inference, trained_params, bounds
