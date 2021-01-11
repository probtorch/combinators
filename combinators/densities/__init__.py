#!/usr/bin/env python3
#
import math
import torch
import operator
from functools import partial, reduce
from torch import Tensor, distributions, Size
from typing import Optional, Dict, Union, Callable

from combinators import Program
from combinators.embeddings import CovarianceEmbedding
from combinators.stochastic import Trace, ImproperRandomVariable, RandomVariable, Provenance
from combinators.types import TraceLike

def typecheck_dims(fn, expected_dims):
    def wrapper(*args, **kwargs):
        tr, out = fn(*args, **kwargs)
        assert len(out.shape) == expected_ndims, f"expected shape should be of length {expected_dims}"
        return tr, out
    return wrapper


class Density(Program):
    """ A program that represents a single unnormalized distribution (and allows for sampling a shape). """

    def __init__(self, name, generator:Callable[[Size], Tensor], log_density_fn:Callable[[Tensor], Tensor]):
        super().__init__()
        self.name = name
        self.generator = generator
        self.log_density_fn = log_density_fn
        self.RandomVariable = ImproperRandomVariable

    def model(self, trace, sample_shape=Size([1, 1])):
        generator = self.generator
        # FIXME: ensure conditioning a program work like this is automated?
        value = trace[self.name].value if self.name in trace else generator(sample_shape=sample_shape) # should be implicit ?

        assert len(value.shape) >= 2, "must have at least sample and output dims"

        rv = self.RandomVariable(fn=generator, log_density_fn=self.log_density_fn, value=value, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace[self.name].value

    def __repr__(self):
        return f'[{self.name}]' + super().__repr__()

class Distribution(Program):
    """ Normalized version of Density but complex class heirarchy is an antipattern, so c/p'd """

    def __init__(self, name, dist):
        super().__init__()
        self.name = name
        self.dist = dist
        self.RandomVariable = RandomVariable

    def model(self, trace, sample_shape=torch.Size([1,1]), validate=True):
        dist = self.dist
        # FIXME: ensure conditioning a program work like this is automated?
        value = trace[self.name].value if self.name in trace else \
            (dist.rsample(sample_shape) if dist.has_rsample else dist.sample(sample_shape))
        if validate and not (len(value.shape) >= 2):
            raise RuntimeError("must have at least sample dim + output dim")
        if not all(map(lambda lr: lr[0] == lr[1], zip(sample_shape, value.shape))):
            adjust_shape = [*sample_shape, *value.shape[len(sample_shape):]]
            value = value.view(adjust_shape)

        rv = self.RandomVariable(dist=dist, value=value, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace[self.name].value

    def __repr__(self):
        return f'[{self.name}]' + super().__repr__()

class Normal(Distribution):
    def __init__(self, loc, scale, name, reparam=True):
        as_tensor = lambda x: x if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float, requires_grad=reparam)

        self.loc = as_tensor(loc)
        self._loc = self.loc.cpu().item()
        self.scale = as_tensor(scale)
        self._scale = self.scale.cpu().item()

        self._dist = distributions.Normal(loc=self.loc, scale=self.scale)
        super().__init__(name, self._dist)

    def __repr__(self):
        return f"Normal(name={self.name}, loc={self._loc}, scale={self._scale})"

    def as_dist(self, as_multivariate=False):
        return self._dist if not as_multivariate else \
            distributions.MultivariateNormal(loc=self._dist.loc.unsqueeze(0), covariance_matrix=torch.eye(1))

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
            g_0:Union[Distribution, Density],
            g_1:Union[Distribution, Density],
            beta:Tensor
    ):
        assert torch.all(beta > 0.) and torch.all(beta < 1.), "tempered densities are β=(0, 1) for clarity/debugging"
        super().__init__(name, self._generator, self.log_density_fn)
        self.beta = beta
        self.g_0 = g_0
        self.g_1 = g_1
        # self.logit = nn.Parameter(torch.logit(beta)) # got rid of parameter_map so this is currently unused...

    def _sample_from(self, g:Union[Distribution, Density, Program], sample_shape:Size=Size([1,1])) -> Tensor:
        tr, out = g(sample_shape=sample_shape)
        return tr[g.name].value

    def _generator(self, sample_shape:Size=Size([1,1])) -> Tensor:
        with torch.no_grad(): # you can't learn anything about this density
            t, g_0, g_1 = self.beta, self.g_0, self.g_1
            t0 = self._sample_from(g_0, sample_shape=sample_shape)*(1-t)
            t1 = self._sample_from(g_1, sample_shape=sample_shape)*t
            return t0 + t1

    def log_density_fn(self, value:Tensor) -> Tensor:
        def log_prob(g, value):
            return g.log_density_fn(value) if isinstance(g, Density) else g.dist.log_prob(value)

        with torch.no_grad(): # you can't learn anything about this density
            t, g_0, g_1 = self.beta, self.g_0, self.g_1

            g_0_log_prob = log_prob(g_0, value)
            g_1_log_prob = log_prob(g_1, value)
            return g_0_log_prob*(1-t) + g_1_log_prob*t

    def __repr__(self):
        return f"β=1/{int((1/self.beta.item())+0.0001)}" + super().__repr__()


class GMM(Density):
    def __init__(self, locs, covs, name="GMM"):
        assert len(locs) == len(covs)
        self.K = K = len(locs)
        super().__init__(name, self._generator, self.log_density_fn)
        self.components = [distributions.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k]) for k in range(K)]
        self.assignments = distributions.Categorical(torch.ones(K)) # take this off the trace
        # self.assignments = Categorical("assignments", probs=torch.ones(K))

    def _generator(self, sample_shape=torch.Size([1])):
        # NOTE: no trace being used here
        zs = self.assignments.sample(sample_shape=sample_shape)

        # trace.update(a_trace)
        cluster_shape = (1, *zs.shape[1:-1])
        xs = []
        values, indicies = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            x_k = self.components[k].sample(sample_shape=(n_k, *zs.shape[1:-1]))
            xs.append(x_k)

        xs = torch.cat(xs)
        return xs[indicies]

    def log_density_fn(self, values):
        return reduce(operator.add, map(lambda k: self.components[k].log_prob(values), range(self.K)))

class RingGMM(GMM):
    def __init__(self, name="RingGMM", scale=10, count=8):
        angles = list(range(0, 360, 360//count))[:count] # integer division may give +1
        position = lambda radians: [math.cos(radians), math.sin(radians)]
        locs = torch.tensor([position(a*math.pi/180) for a in angles]) * scale
        covs = [torch.eye(2) for _ in range(count)]
        super().__init__(name=name, locs=locs, covs=covs)
