#!/usr/bin/env python3

import torch
import combinators.trace.utils as trace_utils
from combinators.tensor.utils import autodevice
from combinators.stochastic import Provenance
from torch import nn
from torch import Tensor
from combinators.program import Program
from combinators.nnets import LinearMap
from combinators.embeddings import CovarianceEmbedding


class ConditionalNormal(Program):
    def __init__(self, ext_from:str, ext_to:str, net):
        super().__init__()
        self.net = net
        self.ext_from = ext_from
        self.ext_to = ext_to

    def model(self, trace, c, sample_dims=None):
        assert isinstance(c, dict) and self.ext_from in c

        mu = self.net(c[self.ext_from].detach())

        value = trace.normal(loc=mu,
                            scale=torch.ones_like(mu, device=mu.device),
                            value=trace[self.ext_to].value if self.ext_to in trace else None,
                            name=self.ext_to)

        return {self.ext_to: value}

    def __repr__(self):
        return f'ext_to={self.ext_to}:' + super().__repr__()


class ConditionalNormalLinear(ConditionalNormal):
    def __init__(self, ext_from, ext_to, device=None):
        super().__init__(ext_from, ext_to, LinearMap(dim=1).to(autodevice(device)))

    def weight(self):
        return self.net.weight()

    def bias(self):
        return self.net.bias()

class ConditionalMultivariateNormal(Program):
    def __init__(
            self,
            ext_from:str,
            ext_to:str,
            loc:Tensor,
            cov:Tensor,
            net:nn.Module,
            embedding_dim:int=2,
            cov_embedding:CovarianceEmbedding=CovarianceEmbedding.SoftPlusDiagonal,
            learn_cov:bool=True
        ):
        super().__init__()
        self.ext_from = ext_from
        self.ext_to = ext_to
        self.dim_in = 2
        self.cov_dim = cov.shape[0]
        self.cov_embedding = cov_embedding
        self.learn_cov = learn_cov

        if learn_cov:
            self.register_parameter(self.cov_embedding.embed_name, nn.Parameter(self.cov_embedding.embed(cov, embedding_dim)))

        self.net = net
        try:
            self.net.initialize_(torch.zeros_like(loc), self.cov_embedding.embed(cov, embedding_dim))
        except:
            pass

    def model(self, trace, c, *shared):
        try:
            assert isinstance(c, dict) and self.ext_from in c
        except:
            breakpoint();

        if self.learn_cov:
            mu, cov_emb = self.net(c[self.ext_from].detach())
            cov = self.cov_embedding.unembed(cov_emb, self.cov_dim)
        else:
            mu = self.net(c[self.ext_from].detach())
            cov = torch.eye(self.cov_dim, device=mu.device)

        value = trace.multivariate_normal(
            loc=mu,
            covariance_matrix=cov,
            value=trace[self.ext_to].value if self.ext_to in trace else None,
            name=self.ext_to)

        return {self.ext_to: value}

class ConditionalMultivariateNormalLinear(ConditionalMultivariateNormal):
    def __init__(self, ext_from:str, ext_to:str, loc:Tensor, cov:Tensor):
        super().__init__(ext_from, ext_to, loc, cov, LinearMap(dim=2), learn_cov=False)

    def weight(self):
        return self.net.weight()

    def bias(self):
        return self.net.bias()
