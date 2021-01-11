#!/usr/bin/env python3

import torch
from torch import nn
from torch import Tensor
from combinators.kernel import Kernel
from combinators.nnets import LinearMap
from combinators.embeddings import CovarianceEmbedding


class NormalKernel(Kernel):
    def __init__(self, ext_name, net):
        super().__init__()
        self.net = net
        self.ext_name = ext_name

    def apply_kernel(self, trace, cond_trace, cond_output, sample_dims=None):
        # TODO: super annoying... I will just assume there is always a sample dimension and will need to add some more guardrails
        # if sample_dims is not None:
        #     if len(cond_output.shape) == 1:
        #         # reshape
        #         with_samples_shape = [*cond_output.shape[:sample_dims+1], 1, *cond_output.shape[sample_dims+1:]]
        #         cond_output = cond_output.view(with_samples_shape)
        #     # breakpoint();
        #     if cond_output.shape[0] == 1 and len(cond_output.shape) == 2:
        #         cond_output = cond_output.T
        #     else:
        #         pass
        sample_shape = cond_output.shape
        if sample_dims is not None and cond_output.shape[0] == 1 and len(cond_output.shape) == 2:
            cond_output = cond_output.T

        mu = self.net(cond_output.detach()).view(sample_shape)

        return trace.normal(loc=mu,
                            scale=torch.ones_like(mu, device=mu.device),
                            value=cond_trace[self.ext_name].value if self.ext_name in cond_trace else None, # this could _and should_ be automated
                            name=self.ext_name)

    def __repr__(self):
        return f'ext={self.ext_name}:' + super().__repr__()

    def weight(self):
        return self.net.weight()

    def bias(self):
        return self.net.bias()

class NormalLinearKernel(NormalKernel):
    def __init__(self, ext_name):
        super().__init__(ext_name, LinearMap(dim=1))

class MultivariateNormalKernel(Kernel):
    def __init__(
            self,
            ext_name:str,
            loc:Tensor,
            cov:Tensor,
            net:nn.Module,
            embedding_dim:int=2,
            cov_embedding:CovarianceEmbedding=CovarianceEmbedding.SoftPlusDiagonal,
            learn_cov:bool=True
        ):
        super().__init__()
        self.ext_name = ext_name
        self.dim_in = 2
        self.cov_dim = cov.shape[0]
        self.cov_embedding = cov_embedding
        self.learn_cov = learn_cov

        if learn_cov:
            self.register_parameter(self.cov_embedding.embed_name, nn.Parameter(self.cov_embedding.embed(cov, embedding_dim)))

        self.net = net
        try:
            # FIXME: A bit of a bad, legacy assumption
            self.net.initialize_(loc, getattr(self, self.cov_embedding.embed_name)) # we don't have cov_embed...
        except:
            pass

    def apply_kernel(self, trace, cond_trace, cond_output, sample_dims=None):
        # sample_shape = cond_output.shape
        # if sample_dims is not None and cond_output.shape[0] == 1 and len(cond_output.shape) == 2:
        #     cond_output = cond_output.T

        # mu, cov_emb = self.net(cond_output.detach()).view(sample_shape)
        if self.learn_cov:
            mu, cov_emb = self.net(cond_output.detach())
            cov = self.cov_embedding.unembed(getattr(self, self.cov_embedding.embed_name), self.cov_dim)
        else:
            mu = self.net(cond_output.detach())
            cov = torch.eye(self.cov_dim, device=mu.device)
        return trace.multivariate_normal(loc=mu,
                                         covariance_matrix=cov,
                                         value=cond_trace[self.ext_name].value if self.ext_name in cond_trace else None,
                                         name=self.ext_name)

class MultivariateNormalLinearKernel(MultivariateNormalKernel):
    def __init__(self, ext_name:str, loc:Tensor, cov:Tensor):
        super().__init__(ext_name, loc, cov, LinearMap(dim=2), learn_cov=False)
