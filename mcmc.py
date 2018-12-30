#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
import torch

import combinators
import trace_tries
import utils

class MHMove(combinators.InferenceSampler):
    def __init__(self, sampler, moves=1):
        super(MHMove, self).__init__(sampler)
        self._args = ()
        self._kwargs = {}
        self._moves = moves

    def sample_prehook(self, trace, *args, **kwargs):
        self._args = args
        self._kwargs = {k: v for (k, v) in kwargs.items() if k != 'trace'}
        return trace, args, kwargs

    def propose(self, results, trace):
        raise NotImplementedError()

    def sample_hook(self, results, trace):
        marginal = trace.marginal_log_likelihood()
        for _ in range(self._moves):
            candidate, candidate_trace, move_candidate, move_current =\
                self.propose(results, trace)
            candidate_marginal = candidate_trace.marginal_log_likelihood()
            mh_ratio = (candidate_marginal - move_candidate) -\
                       (marginal - move_current)
            log_alpha = torch.min(torch.zeros(1), mh_ratio)
            if torch.bernoulli(torch.exp(log_alpha)) == 1:
                results = candidate
                trace = candidate_trace
                marginal = candidate_marginal
        self._args = ()
        self._kwargs = {}
        return results, trace

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
