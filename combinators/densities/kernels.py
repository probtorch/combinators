#!/usr/bin/env python3

import torch
import combinators.trace.utils as trace_utils
from combinators.tensor.utils import autodevice
from torch import nn
from torch import Tensor
from combinators.kernel import Kernel
from combinators.nnets import LinearMap
from combinators.embeddings import CovarianceEmbedding


class IdentityKernel(Kernel):
    def __init__(self, name, extends:bool=True):
        super().__init__()
        assert extends, "TODO: a true identity kernel -- maybe after NVI"
        self.ext_name = name

    def apply_kernel(self, trace, cond_trace, cond_output, sample_dims=None):
        assert len(cond_trace) == 1
        cond_addr = list(cond_trace.keys())[0]
        newrv = trace_utils.copyrv(cond_trace[cond_addr], requires_grad=True)
        trace.append(newrv, name=self.ext_name)
        return cond_output


class NormalKernel(Kernel):
    def __init__(self, ext_from:str, ext_to:str, net):
        super().__init__()
        self.net = net
        self.ext_from = ext_from
        self.ext_to = ext_to

    def apply_kernel(self, trace, cond_trace, cond_output, sample_dims=None):
        # def samples_first(aten, sample_dims):
        #     shape = aten.shape
        #     sdim = sample_dims
        #     sample_cardinality = shape[sdim]
        #     dims_left = shape[:sdim]
        #     dims_right = shape[sdim+1:]
        #     return aten.view([sample_cardinality, *dims_left, *dims_right])
        # cond_value = samples_first(cond_trace[self.ext_from].value, sample_dims)
        cond_value = cond_trace[self.ext_from].value
        mu = self.net(cond_value.detach())

        return trace.normal(loc=mu,
                            scale=torch.ones_like(mu, device=mu.device),
                            value=cond_trace[self.ext_to].value if self.ext_to in cond_trace else None, # this could _and should_ be automated
                            name=self.ext_to)

    def __repr__(self):
        return f'ext_to={self.ext_to}:' + super().__repr__()


class NormalLinearKernel(NormalKernel):
    def __init__(self, ext_from, ext_to, device=None):
        super().__init__(ext_from, ext_to, LinearMap(dim=1).to(autodevice(device)))

    def weight(self):
        return self.net.weight()

    def bias(self):
        return self.net.bias()

class MultivariateNormalKernel(Kernel):
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
            mu, cov_emb = self.net(cond_trace[self.ext_from].value.detach())
            cov = self.cov_embedding.unembed(getattr(self, self.cov_embedding.embed_name), self.cov_dim)
        else:
            mu = self.net(cond_trace[self.ext_from].value.detach())
            cov = torch.eye(self.cov_dim, device=mu.device)
        return trace.multivariate_normal(loc=mu,
                                         covariance_matrix=cov,
                                         value=cond_trace[self.ext_to].value if self.ext_to in cond_trace else None,
                                         name=self.ext_to)

class MultivariateNormalLinearKernel(MultivariateNormalKernel):
    def __init__(self, ext_from:str, ext_to:str, loc:Tensor, cov:Tensor):
        super().__init__(ext_from, ext_to, loc, cov, LinearMap(dim=2), learn_cov=False)

    def weight(self):
        return self.net.weight()

    def bias(self):
        return self.net.bias()
