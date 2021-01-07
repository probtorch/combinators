import math
import torch
from functools import partial
from torch import Tensor, distributions, Size, nn
from combinators.embeddings import CovarianceEmbedding
from combinators.stochastic import Trace, ImproperRandomVariable, RandomVariable, Provenance
from combinators.types import TraceLike
from combinators.densities import MultivariateNormal, Categorical
from combinators import Kernel, Program
from abc import ABC
from typing import Optional, Dict, Union, Callable


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
        # FIXME: incorrect codomain
        log_probs = torch.cat([tr[k].log_prob for k in [f'g_{i}' for i in range(self.K)]])
        return log_probs[ixs]


class RingGMM(GMM):
    def __init__(self, name="RingGMM", scale=10, count=8):
        angles = list(range(0, 360, 360//count))[:count] # integer division may give +1
        position = lambda radians: [math.cos(radians), math.sin(radians)]
        locs = torch.tensor([position(a*math.pi/180) for a in angles]) * scale
        covs = [torch.eye(2) for _ in range(count)]
        super().__init__(name=name, locs=locs, covs=covs)


class ResMLPJ(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, with_cov_embedding=False, initialize=None):
        assert initialize is None or initialize in ['truncated_normal']
        self._initialize_type = initialize
        super().__init__()
        self.with_cov_embedding = with_cov_embedding # TODO: ask heiko why this is ignored
        self.initialize = initialize
        self.joint = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = torch.nn.Sequential(torch.nn.Linear(dim_hidden, dim_out))
        self.cov = torch.nn.Sequential(torch.nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        y = self.joint(x)
        mu = self.mu(y) + x # FIXME: + x ?
        cov_emb = self.cov(y)
        return mu, cov_emb

    def initialize_(self, loc_offset, cov_emb):
        if self._initialize_type is None or self._initialize_type in ['truncated_normal']:
            initer = torch.nn.init.zeros_
        elif self._initialize_type == 'truncated_normal':
            initer = lambda aten: torch.nn.init.normal_(aten, mean=0., std=0.01)
        else:
            raise TypeError()

        _ = [initer(ten) for ten in [self.cov[0].weight, self.cov[0].bias]]
        if self.with_cov_embedding:
            self.cov[0].bias.data.add_(cov_emb)
        else:
            self.mu[0].bias.data.add_(loc_offset)


class MultivariateNormalKernel(Kernel):
    def __init__(
            self,
            ext_name:str,
            loc:Tensor,
            cov:Tensor,
            dim_hidden:int=32,
            embedding_dim:int=2,
            cov_embedding:CovarianceEmbedding=CovarianceEmbedding.SoftPlusDiagonal,
        ):
        super().__init__()
        self.dim_in = 2
        self.cov_shape = cov.shape
        self.cov_embedding = cov_embedding
        self.register_parameter(self.cov_embedding.embed_name, nn.Parameter(self.cov_embedding.embed(cov, embedding_dim)))

        self.net = ResMLPJ(dim_in=self.dim_in, dim_hidden=dim_hidden, dim_out=embedding_dim, with_cov_embedding=False)
        self.net.initialize_(loc, getattr(self, self.cov_embedding.embed_name)) # we don't have cov_embed...

    def apply_kernel(self, trace, cond_trace, cond_output):
        mu, cov_emb = self.net(cond_output.detach())
        cov = self.cov_embedding.unembed(getattr(self, self.cov_embedding.embed_name), self.cov_shape)
        return trace.multivariate_normal(loc=mu, covariance_matrix=cov, name=self.ext_name)
