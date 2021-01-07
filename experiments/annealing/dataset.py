#!/usr/bin/env python3
import math
import torch
from torch import Tensor, distributions
from combinators.stochastic import Trace, RandomVariable, Provenance
from combinators import Program
from abc import ABC

class Distribution(Program):
    def __init__(self, name, dist):
        super().__init__()
        self.name = name
        self.dist = dist

    def model(self, trace, sample_shape=torch.Size([1])):
        dist = self.dist
        value = dist.rsample(sample_shape) if dist.has_rsample else dist.sample(sample_shape)
        rv = RandomVariable(dist=dist, value=value, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return value

class MultivariateNormal(Distribution):
    def __init__(self, loc, cov, name): #, num_samples=100):
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, dtype=torch.float)
        self.cov = cov if isinstance(cov, Tensor) else torch.tensor(cov, dtype=torch.float)
        dist = distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
        super().__init__(name, dist)

class Categorical(Distribution):
    def __init__(self, name, probs=None, logits=None, validate_args=None): #, num_samples=100):
        self.probs = probs
        self.logits = logits
        self.validate_args = validate_args
        super().__init__(name, distributions.Categorical(probs, logits, validate_args))

class GMM(Program):
    def __init__(self, locs, covs, name="GMM"):
        super().__init__()
        assert len(locs) == len(covs)
        self.K = K = len(locs)
        self.components = [MultivariateNormal(name=f'g_{k}', loc=locs[k], cov=covs[k]) for k in range(K)]
        self.assignments = Categorical("assignments", probs=torch.ones(K))

    def model(self, trace, sample_shape=torch.Size([1]), with_indicies=False):
        a_trace, zs = self.assignments(sample_shape=sample_shape)
        trace.update(a_trace)
        cluster_shape = (1, *zs.shape[1:-1])
        xs = []
        values, indicies = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            g_tr, x_k = self.components[k](sample_shape=(n_k, *zs.shape[1:-1]))
            trace.update(g_tr)
            xs.append(x_k)
        xs = torch.cat(xs)

        if with_indicies:
            return (xs[indicies], zs, indicies)
        else:
            return (xs[indicies], zs)

    def log_probs(self, tr, ixs):
        log_probs = torch.cat([tr[k].log_prob for k in [f'g_{i}' for i in range(self.K)]])
        return log_probs[ixs]


class RingGMM(GMM):
    def __init__(self, name="RingGMM", scale=10, count=8):
        angles = list(range(0, 360, 360//count))[:count]
        degrees = lambda angle: [math.cos(angle), math.sin(angle)]
        locs = torch.tensor([degrees(a*math.pi/180) for a in angles]) * scale
        covs = [torch.eye(2) for _ in range(count)]
        super().__init__(name=name, locs=locs, covs=covs)

