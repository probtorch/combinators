#!/usr/bin/env python3
import torch
from abc import ABC
from functools import partial
from torch import Tensor, distributions, Size
from typing import Optional, Dict, Union, Callable

from combinators import Program
from combinators.embeddings import CovarianceEmbedding
from combinators.stochastic import Trace, ImproperRandomVariable, RandomVariable, Provenance
from combinators.types import TraceLike


class Density(Program):
    """ A program that represents a single unnormalized distribution (and allows for sampling a shape). """

    def __init__(self, name, generator:Callable[[Size], Tensor]):
        super().__init__()
        self.name = name
        self.generator = generator
        self.RandomVariable = ImproperRandomVariable

    def model(self, trace, sample_shape=Size([1, 1])):
        generator = self.generator
        value=generator(sample_shape=sample_shape)
        assert len(value.shape) >= 2, "must have at least sample and output dims"

        rv = self.RandomVariable(fn=generator, value=value, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace[self.name].value

class Distribution(Program):
    """ Normalized version of Density but complex class heirarchy is an antipattern, so c/p'd """

    def __init__(self, name, dist):
        super().__init__()
        self.name = name
        self.dist = dist
        self.RandomVariable = RandomVariable

    def model(self, trace, sample_shape=torch.Size([1, 1])):
        dist = self.dist
        value = dist.rsample(sample_shape) if dist.has_rsample else dist.sample(sample_shape)
        assert len(value.shape) >= 2, "must have at least sample and output dims"

        rv = self.RandomVariable(dist=dist, value=value, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace[self.name].value

class MultivariateNormal(Distribution):
    def __init__(self, loc, cov, name, reparam=True):
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc, dtype=torch.float, requires_grad=reparam)
        self.cov = cov if isinstance(cov, Tensor) else torch.tensor(cov, dtype=torch.float, requires_grad=reparam)
        dist = distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
        super().__init__(name, dist)

class Categorical(Distribution):
    def __init__(self, name, probs=None, logits=None, validate_args=None): #, num_samples=100):
        self.probs = probs
        self.logits = logits
        self.validate_args = validate_args
        super().__init__(name, distributions.Categorical(probs, logits, validate_args))

class Tempered(Density):
    def __init__(
            self,
            name,
            g_0:Union[Distribution, Density, Program],
            g_1:Union[Distribution, Density, Program],
            beta:Tensor
    ):
        assert torch.all(beta > 0.) and torch.all(beta < 1.), "tempered densities are β=(0, 1) for clarity/debugging"
        super().__init__(name, self._generator)
        self.beta = beta
        self.g_0 = g_0
        self.g_1 = g_1
        # self.logit = nn.Parameter(torch.logit(beta)) # got rid of parameter_map so this is currently unused...

    def _sample_from(self, sample_shape:Size, g:Union[Distribution, Density, Program]) -> Tensor:
       if isinstance(g, (Distribution, Density)):
           tr, out = g(sample_shape)
       else:
           try:
               # need to be more opinionated here : /
               tr, out = g(sample_shape=sample_shape)
           except:
               tr, out = g()
               assert tr[self.name].value.shape == sample_shape
       return tr[self.name].value

    def _generator(self, sample_shape:Size) -> Tensor:
        with torch.no_grad(): # you can't learn anything about this density
            t, g_0, g_1 = self.beta, self.g_0, self.g_1
            sample_from = partial(self._sample_from, sample_shape)
            # breakpoint();
            return sample_from(g_0)*(1-t) + sample_from(g_1)*t


    def log_probs(self, values:TraceLike) -> Dict[str, Tensor]:
        with torch.no_grad(): # you can't learn anything about this density
            t, g_0, g_1 = self.beta, self.g_0, self.g_1
            g_0_log_probs = g_0.log_probs(values)
            g_1_log_probs = g_1.log_probs(values)
            return {self.name: g_0_log_probs[self.name]*(1-t) + g_1_log_probs[self.name]*t}
