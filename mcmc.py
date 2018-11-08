#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
from probtorch.util import log_mean_exp
import torch

import combinators

class Collection(combinators.ModelSampler):
    def __init__(self, sampler, num_samples, acceptance):
        super(Collection, self).__init__()
        self._sampler = sampler
        self._num_samples = num_samples
        self._acceptance = acceptance

    @property
    def name(self):
        return 'Collection(%s, %d)' % (self.sampler.name, self.num_samples)

    @property
    def sampler(self):
        return self._sampler

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def acceptance(self):
        return self._acceptance

    def _forward(self, *args, **kwargs):
        samples = []
        trace = kwargs.pop('trace')
        while len(samples) < self.num_samples:
            kwargs['trace'] = trace.extract(self.sampler.name)
            sample, sample_trace = self._sampler(*args, **kwargs)
            if self._acceptance(sample, sample_trace):
                trace.insert(str(len(samples)) + '/' + self.sampler.name,
                             sample_trace)
                samples.append(sample)
        return samples, trace

def independent_mh(model, num_samples):
    weight = 0.0
    def acceptance(_, trace):
        nonlocal weight
        candidate = trace.marginal_log_likelihood()
        alpha = torch.min(torch.zeros(1), candidate - weight)
        result = weight == 0.0 or torch.bernoulli(torch.exp(alpha)) == 1
        if result:
            weight = candidate
        return result
    return Collection(model, num_samples, acceptance)
