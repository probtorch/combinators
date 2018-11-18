#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
import torch

import combinators

class MHMove(combinators.InferenceSampler):
    def propose(self, results, trace):
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
