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
    def __init__(self, sampler):
        super(MHMove, self).__init__(sampler)
        self._args = ()
        self._kwargs = {}

    def sample_prehook(self, trace, *args, **kwargs):
        self._args = args
        self._kwargs = {k: v for (k, v) in kwargs.items() if k != 'trace'}
        return trace, args, kwargs

    def propose(self, results, trace):
        self._args = ()
        self._kwargs = {}
        raise NotImplementedError()

    def sample_hook(self, results, trace):
        marginal = trace.marginal_log_likelihood()
        candidate, candidate_trace, move_candidate, move_current =\
            self.propose(results, trace)
        candidate_marginal = candidate_trace.marginal_log_likelihood()
        mh_ratio = (candidate_marginal - move_candidate) -\
                   (marginal - move_current)
        log_alpha = torch.min(torch.zeros(1), mh_ratio)
        if torch.bernoulli(torch.exp(log_alpha)) == 1:
            return candidate, candidate_trace
        return results, trace

class LightweightMH(MHMove):
    def propose(self, results, trace):
        rv_index = np.random.randint(len(trace))
        candidate = trace[0:rv_index]
        move_current = trace[rv_index].log_prob
        candidate.variable(trace[rv_index].dist, name=trace[rv_index].name,
                           value=utils.try_rsample(trace[rv_index].dist))
        move_candidate = candidate[-1].log_prob
        results, candidate = self.sampler(*self._args, **self._kwargs,
                                          trace=candidate)
        self._args = ()
        self._kwargs = {}
        return results, candidate, move_candidate, move_current
