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
    def __init__(self, sampler, moves=1):
        super(MHMove, self).__init__(sampler)
        self._moves = moves

    def propose(self, results, trace, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        zs, xi, w = self.sampler(*args, **kwargs)
        original_traces = xi
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)
        marginal = utils.marginalize_all(w)
        for _ in range(self._moves):
            zsq, xiq, wq, move_proposed, move_current =\
                self.propose(zs, xi, original_traces, *args, **kwargs)
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

class LightweightMH(MHMove):
    def propose(self, results, trace):
        rv_index = np.random.randint(len(trace))
        while trace[rv_index].observed:
            rv_index = np.random.randint(len(trace))
        candidate = trace[0:rv_index]
        move_current = utils.marginalize_all(trace[rv_index].log_prob)
        dist = trace[rv_index].dist
        rv = probtorch.RandomVariable(dist, utils.try_rsample(dist), False)
        candidate[trace.name(rv_index)] = rv
        results, candidate = self.sampler(*self._args, **self._kwargs,
                                          trace=candidate)
        move_candidate = utils.marginalize_all(rv.log_prob)
        return results, candidate, move_candidate, move_current

def resample_move_smc(sampler, particle_shape, initializer=None, moves=1,
                      mcmc=LightweightMH):
    resampler_mover = mcmc(importance.ImportanceResampler(sampler,
                                                          particle_shape),
                           moves)
    return foldable.Foldable(resampler_mover, initializer=initializer)

def reduce_resample_move_smc(stepwise, particle_shape, step_generator,
                             initializer=None, moves=1, mcmc=LightweightMH):
    rmsmc_foldable = resample_move_smc(stepwise, particle_shape,
                                       initializer=initializer, moves=moves,
                                       mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
