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

    def propose(self, results, graph, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        zs, xi, w = self.sampler(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)
        marginal = utils.marginalize_all(w)
        for _ in range(self._moves):
            zsq, xiq, wq, move_proposed, move_current =\
                self.propose(zs, xi, *args, **kwargs)
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
    def propose(self, results, graph, *args, **kwargs):
        t = np.random.randint(len(graph))
        sampled = [k for k in graph[t] if not graph[t][k].observed]
        while not sampled:
            t = np.random.randint(len(graph))
            sampled = [k for k in graph[t] if not graph[t][k].observed]

        address = sampled[np.random.randint(len(sampled))]
        candidate_trace = utils.slice_trace(graph[t], address)
        candidate_graph = graph.graft(t, candidate_trace)

        zs, xi, w = self.sampler.cond(candidate_graph)(*args, **kwargs)

        predicate = lambda k, v: not v.observed
        move_candidate = utils.marginalize_all(xi.log_joint())
        move_candidate += torch.log(torch.tensor(float(
            xi.num_variables(predicate=predicate)
        )))
        move_current = utils.marginalize_all(graph.log_joint())
        move_current += torch.log(torch.tensor(float(
            graph.num_variables(predicate=predicate)
        )))

        return zs, xi, w, move_candidate, move_current

    def walk(self, f):
        return LightweightMH(self.sampler.walk(f), self._moves)

    def cond(self, qs):
        return LightweightMH(self.sampler.cond(qs), self._moves)

def resample_move_smc(sampler, initializer=None, moves=1, mcmc=LightweightMH):
    fold = foldable.Foldable(sampler, initializer=initializer)
    return mcmc(fold.walk(importance.ImportanceResampler), moves)

def reduce_resample_move_smc(stepwise, step_generator, initializer=None,
                             moves=1, mcmc=LightweightMH):
    rmsmc_foldable = resample_move_smc(stepwise, initializer=initializer,
                                       moves=moves, mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
