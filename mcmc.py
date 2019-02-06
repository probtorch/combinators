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
    def propose(self, results, traces, originals, *args, **kwargs):
        t = np.random.randint(len(traces))
        sampled = list(originals[t].sampled())
        while not sampled:
            t = np.random.randint(len(traces))
            sampled = list(originals[t].sampled())

        address = sampled[np.random.randint(len(sampled))]
        original = traces[t][address]

        candidate = utils.slice_trace(traces[t], address)
        move_current = utils.marginalize_all(original.log_prob)
        dist = original.dist
        rv = probtorch.RandomVariable(dist, utils.try_rsample(dist),
                                      probtorch.stochastic.Provenance.SAMPLED)
        candidate[address] = rv
        candidates = traces.graft(t, candidate)
        zs, xi, w = self.sampler.cond(candidates)(*args, **kwargs)
        move_candidate = utils.marginalize_all(rv.log_prob)
        return zs, xi, w, move_candidate, move_current

    def walk(self, f):
        return LightweightMH(self.sampler.walk(f), self._moves)

    def cond(self, qs):
        return LightweightMH(self.sampler.cond(qs), self._moves)

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
