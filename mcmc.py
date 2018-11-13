#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
import torch

import combinators

def stack_samples(samples):
    for i in len(samples[0]):
        yield torch.stack([sample[i] for sample in samples], dim=0)

class Collection(combinators.InferenceSampler):
    def __init__(self, sampler, num_samples):
        super(Collection, self).__init__(sampler)
        self._num_samples = num_samples

    @property
    def num_samples(self):
        return self._num_samples

    def accept(self, results, trace):
        raise NotImplementedError()

    def sample_prehook(self, trace, *args, **kwargs):
        return trace, args, kwargs

    def forward(self, *args, **kwargs):
        samples = []
        trace = kwargs.pop('trace')
        while len(samples) < self.num_samples:
            kwargs['trace'] = trace.extract(self.sampler.name)
            sample, sample_trace = self.sampler(*args, **kwargs)
            if self.accept(sample, sample_trace):
                trace.insert(str(len(samples)) + '/' + self.sampler.name,
                             sample_trace)
                samples.append(sample)
        return samples, trace

    def sample_hook(self, results, trace):
        return list(stack_samples(results)), trace

class IndependentMH(Collection):
    def __init__(self, sampler, num_samples):
        super(IndependentMH, self).__init__(sampler, num_samples)
        self._weight = 0.0

    def accept(self, results, trace):
        candidate = trace.marginal_log_likelihood()
        alpha = torch.min(torch.zeros(1), candidate - self._weight)
        result = self._weight == 0.0 or torch.bernoulli(torch.exp(alpha)) == 1
        if result:
            self._weight = candidate
        return result
