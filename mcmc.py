#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
import torch

import combinators
import foldable
import importance
import utils

class MHMove(combinators.Inference):
    def __init__(self, sampler, kernel, moves=1):
        super(MHMove, self).__init__(sampler)
        assert isinstance(kernel, combinators.TransitionKernel)
        self.add_module('kernel', kernel)
        self._moves = moves

    def forward(self, *args, **kwargs):
        zs, xi, w = self.sampler(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)
        marginal = utils.marginalize_all(w)
        for _ in range(self._moves):
            zsq, xiq, wq, move_proposed, move_current =\
                self.kernel(zs, xi, w, *args, **kwargs)
            marginal_q = utils.marginalize_all(wq)
            mh_ratio = (marginal_q - move_proposed) - (marginal - move_current)
            log_alpha = torch.min(torch.zeros(mh_ratio.shape), mh_ratio)
            if torch.bernoulli(torch.exp(log_alpha)) == 1:
                zs = zsq
                xi = xiq
                w = wq
        if not multiple_zs:
            zs = zs[0]
        return zs, xi, w

    def walk(self, f):
        return MHMove(self.sampler.walk(f), self.kernel, self._moves)

    def cond(self, qs):
        return MHMove(self.sampler.cond(qs), self.kernel, self._moves)

class LightweightKernel(combinators.TransitionKernel):
    def __init__(self, prior):
        super(combinators.TransitionKernel, self).__init__(prior.batch_shape)
        assert isinstance(prior, combinators.Sampler)
        self.add_module('prior', prior)

    def forward(self, zs, xi, w, *args, **kwargs):
        sampled = []
        while not sampled:
            t = np.random.randint(len(xi))
            sampled = [k for k in xi[t] if not xi[t][k].observed]

        address = sampled[np.random.randint(len(sampled))]
        candidate_trace = utils.slice_trace(xi[t], address)
        candidate_graph = xi.graft(t, candidate_trace)

        zsq, xiq, wq = self.prior.cond(candidate_graph)(*args, **kwargs)

        predicate = lambda k, v: not v.observed
        move_candidate = utils.marginalize_all(xiq.log_joint())
        move_candidate += torch.log(torch.tensor(float(
            xiq.num_variables(predicate=predicate)
        )))
        move_current = utils.marginalize_all(xi.log_joint())
        move_current += torch.log(torch.tensor(float(
            xi.num_variables(predicate=predicate)
        )))

        return zsq, xiq, wq, move_candidate, move_current

    def walk(self, f):
        return LightweightKernel(self.prior.walk(f))

    def cond(self, qs):
        return LightweightKernel(self.sampler.cond(qs[self.name:]))

    @property
    def name(self):
        return 'LightweightKernel(%s)' % self.prior.name

def lightweight_mh(sampler, moves=1):
    return MHMove(sampler, LightweightKernel(sampler), moves=moves)

def resample_move_smc(sampler, initializer=None, moves=1, mcmc=lightweight_mh):
    fold = foldable.Foldable(sampler, initializer=initializer)
    inference = lambda m: mcmc(importance.ImportanceResampler(m), moves)
    return fold.walk(inference)

def reduce_resample_move_smc(stepwise, step_generator, initializer=None,
                             moves=1, mcmc=lightweight_mh):
    rmsmc_foldable = resample_move_smc(stepwise, initializer=initializer,
                                       moves=moves, mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
