#!/usr/bin/env python3
import math
import torch
import operator
from functools import partial, reduce
from torch import Tensor, distributions, Size, nn
import torch.distributions as D
from typing import Optional, Dict, Union, Callable
from combinators.tensor.utils import kw_autodevice, autodevice
import combinators.trace.utils as trace_utils

from combinators import Program
from combinators.embeddings import CovarianceEmbedding
from combinators.stochastic import Trace, ImproperRandomVariable, RandomVariable, Provenance
from combinators.types import TraceLike

class Distribution(Program):
    """ Normalized version of Density but trying to limit overly-complex class heirarchies """

    def __init__(self, name:str, dist:distributions.Distribution):
        super().__init__()
        self.name = name
        self.dist = dist
        self.RandomVariable = RandomVariable

    def model(self, trace, c, sample_shape=torch.Size([1,1])):
        dist = self.dist
        value, provenance = trace_utils.maybe_sample(trace, sample_shape)(dist, self.name)

        rv = self.RandomVariable(dist=dist, value=value, provenance=provenance) # <<< rv.log_prob = dist.log_prob(value)
        trace.append(rv, name=self.name)
        return {self.name: rv.value}

    def __repr__(self):
        return f'Distribution[name={self.name}; dist={repr(self.dist)}]'

class Normal(Distribution):
    def __init__(self, loc, scale, name, reparam=True, device=None):
        as_tensor = lambda x: x.to(autodevice(device)) if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float, requires_grad=reparam, **kw_autodevice(device))

        self.loc, self.scale = as_tensor(loc), as_tensor(scale)
        self._loc, self._scale = self.loc.cpu().item(), self.scale.cpu().item()

        self._dist = distributions.Normal(loc=self.loc, scale=self.scale)
        super().__init__(name, self._dist)

    def as_dist(self, as_multivariate=False):
        return self._dist if not as_multivariate else \
            distributions.MultivariateNormal(loc=self._dist.loc.unsqueeze(0), covariance_matrix=torch.eye(1, device=self._dist.loc.device))

class MultivariateNormal(Distribution):
    def __init__(self, loc, cov, name, reparam=True, device=None):
        as_tensor = lambda x: x.to(autodevice(device)) if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float, requires_grad=reparam, **kw_autodevice(device))
        self.loc, self.cov = as_tensor(loc), as_tensor(cov)
        dist = distributions.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)
        super().__init__(name, dist)

class Categorical(Distribution):
    def __init__(self, name, probs=None, logits=None, validate_args=None): #, num_samples=100):
        self.probs = probs
        self.logits = logits
        self.validate_args = validate_args
        super().__init__(name, distributions.Categorical(probs, logits, validate_args))

class OneHotCategorical(Distribution):
    def __init__(self, name, probs=None, logits=None, validate_args=None):
        self.probs = probs
        self.logits = logits
        self.validate_args = validate_args
        super().__init__(name, distributions.OneHotCategorical(probs, logits, validate_args))


class NormalGamma(Distribution):
    """ The generative model of GMM """
    PRECISIONS = 'precisions'
    MEANS = 'means'

    def __init__(self, mu, nu, alpha, beta, prefix=""):
        self.prefix = '' if len(prefix) == 0 else prefix + "_"
        self.PRECISIONS = self.prefix + NormalGamma.PRECISIONS
        self.MEANS = self.prefix + NormalGamma.MEANS
        super().__init__(dist=None, name=f"{{{self.PRECISIONS}, {self.MEANS}}}")

        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.beta = beta

    def model(self, trace, c, sample_shape=None):
        gamma = D.Gamma(self.alpha, self.beta)
        precisions, provenance = trace_utils.maybe_sample(trace, sample_shape)(gamma, self.PRECISIONS)
        trace.append(RandomVariable(dist=gamma, value=precisions, provenance=provenance), name=self.PRECISIONS)

        normal = D.Normal(self.mu, 1. / (self.nu * precisions).sqrt())
        means, provenance = trace_utils.maybe_sample(trace, None)(normal, self.MEANS)
        trace.append(RandomVariable(dist=normal, value=means, provenance=provenance), name=self.MEANS)

        return (precisions, means)


class Density(Program):
    """ A program that represents a single unnormalized distribution that you can query logprobs on. """

    def __init__(self, name, log_density_fn:Callable[[Tensor], Tensor]):
        super().__init__()
        self.name = name
        self.log_density_fn = log_density_fn

    def model(self, trace, c):
        assert self.name in trace, "an improper RV can only condition on values in an existing trace"
        rv = ImproperRandomVariable(log_density_fn=self.log_density_fn, value=trace[self.name].value, provenance=Provenance.REUSED)
        trace.append(rv, name=self.name)
        return {self.name: rv.value}

    def __repr__(self):
        return f'[{self.name}]' + super().__repr__()

class Tempered(Density):
    def __init__(self, name, d1:Union[Distribution, Density], d2:Union[Distribution, Density], beta:Tensor, optimize=False):
        assert torch.all(beta > 0.) and torch.all(beta < 1.), \
            "tempered densities are β=(0, 1) for clarity. Use model directly for β=0 or β=1"
        super().__init__(name, self.log_density_fn)
        self.beta = beta
        self.density1 = d1
        self.density2 = d2

        if optimize:
            self.logit = nn.Parameter(torch.logit(beta))

    def log_density_fn(self, value:Tensor) -> Tensor:

        def log_prob(g, value):
            # FIXME: technically if we also learn d1 or d2, we must detach parameters first
            assert all([p.requires_grad == False for p in g.parameters()])

            return g.log_density_fn(value) if isinstance(g, Density) else g.dist.log_prob(value)

        t = self.beta
        return log_prob(self.density1, value)*(1-t) + \
               log_prob(self.density2, value)*t


    def __repr__(self):
        return "[β={:.4f}]".format(self.beta.item()) + super().__repr__()


class GMM(Density):
    def __init__(self, locs, covs, name="GMM"):
        assert len(locs) == len(covs)
        self.K = K = len(locs)
        super().__init__(name, self.log_density_fn)
        self.components = [distributions.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k]) for k in range(K)]
        self.assignments = distributions.Categorical(torch.ones(K, device=locs[0].device)) # take this off the trace

    def sample(self, sample_shape=torch.Size([1])):
        """ only used to visualize samples """
        # NOTE: no trace being used here
        trace = Trace()
        zs = self.assignments.sample(sample_shape=sample_shape)

        # trace.update(a_trace)
        cluster_shape = (1, *zs.shape[1:-1])
        xs = []
        values, indicies = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            x_k = self.components[k].sample(sample_shape=(n_k, *zs.shape[1:-1]))
            xs.append(x_k)

        xs = torch.cat(xs)[indicies]

        rv = ImproperRandomVariable(log_density_fn=self.log_density_fn, value=xs, provenance=Provenance.SAMPLED)
        trace.append(rv, name=self.name)
        return trace, xs

    def log_density_fn(self, value): # , log_weights, cond_set, param_set):
        lds = []
        for i, comp in enumerate(self.components):
            ld_i = comp.log_prob(value) # + log_weights[i]
            lds.append(ld_i)
        lds_ = torch.stack(lds, dim=0)
        ld = torch.logsumexp(lds_, dim=0)
        # breakpoint()
        return ld

class RingGMM(GMM):
    def __init__(self, name="RingGMM", loc_scale=5, scale=1, count=8, device=None):
        angles = list(range(0, 360, 360//count))[:count] # integer division may give +1
        position = lambda radians: [math.cos(radians), math.sin(radians)]
        locs = torch.tensor([position(a*math.pi/180) for a in angles], **kw_autodevice(device)) * loc_scale
        covs = [torch.eye(2, **kw_autodevice(device)) * scale for _ in range(count)]
        super().__init__(name=name, locs=locs, covs=covs)

