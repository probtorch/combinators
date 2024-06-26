# Copyright 2021-2024 Northeastern University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
import math
from typing import Union, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, distributions, nn

import combinators.trace.utils as trace_utils
from combinators import (
    Program,
    ImproperRandomVariable,
    RandomVariable,
    autodevice,
    kw_autodevice,
    Provenance,
)


def mk_model(K, optimize_path=False):
    mod = paper_model(K, optimize_path=optimize_path)
    targets, forwards, reverses = [
        [m.to(autodevice()) for m in mod[n]]
        for n in ["targets", "forwards", "reverses"]
    ]
    return targets, forwards, reverses


class ResMLPJ(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.map_joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.map_mu = nn.Sequential(nn.Linear(dim_hidden, dim_out))
        self.map_cov = nn.Sequential(nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        y = self.map_joint(x)
        mu = self.map_mu(y) + x
        cov_emb = self.map_cov(y)
        return torch.cat((mu, cov_emb), dim=-1)

    def initialize_(self, loc_offset, cov_emb):
        nn.init.zeros_(self.map_mu[0].weight)
        nn.init.zeros_(self.map_mu[0].bias)
        self.map_mu[0].bias.data.add_(loc_offset)

        nn.init.zeros_(self.map_cov[0].weight)
        nn.init.zeros_(self.map_cov[0].bias)
        self.map_cov[0].bias.data.add_(cov_emb)


class Density(Program):
    """A program that represents a single unnormalized distribution that you can query logprobs on."""

    def __init__(self, name, log_density_fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.name = name
        self.log_density_fn = log_density_fn

    def model(self, trace, c):
        assert (
            trace._cond_trace is not None and self.name in trace._cond_trace
        ), "an improper RV can only condition on values in an existing trace"
        rv = ImproperRandomVariable(
            log_density_fn=self.log_density_fn,
            value=trace._cond_trace[self.name].value,
            provenance=Provenance.REUSED,
        )
        trace._inject(rv, name=self.name, silent=True)
        return {self.name: rv.value}

    def __repr__(self):
        return f"[{self.name}]" + super().__repr__()


class Distribution(Program):
    """Normalized version of Density but trying to limit overly-complex class heirarchies"""

    def __init__(
        self, name: str, dist: distributions.Distribution, reparameterized: bool
    ):
        super().__init__()
        self.name = name
        self.dist = dist
        self.RandomVariable = RandomVariable
        self.reparameterized = reparameterized

    def model(self, trace, c, sample_shape=torch.Size([1, 1])):
        dist = self.dist

        value, provenance, _ = trace_utils.maybe_sample(
            self._cond_trace, sample_shape, reparameterized=self.reparameterized
        )(dist, self.name)

        rv = self.RandomVariable(
            dist=dist,
            value=value,
            provenance=provenance,
            reparameterized=self.reparameterized,
        )  # <<< rv.log_prob = dist.log_prob(value)
        trace._inject(rv, name=self.name, silent=True)
        return {self.name: rv.value}

    def __repr__(self):
        return f"Distribution[name={self.name}; dist={repr(self.dist)}]"


class FixedMultivariateNormal(Distribution):
    def __init__(self, loc, cov, name, device=None):
        dist = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
        super().__init__(name, dist, reparameterized=False)
        self.loc, self.cov = loc, cov
        self.loc.requires_grad_(False)
        self.cov.requires_grad_(False)


class ConditionalMultivariateNormal(Program):
    def __init__(
        self,
        ext_from: str,
        ext_to: str,
        loc: Tensor,
        cov: Tensor,
        net,
        reparameterized,
    ):
        super().__init__()
        self.ext_from = ext_from
        self.ext_to = ext_to
        self.dim_in = 2
        self.cov_dim = cov.shape[0]

        self.cov_emb = cov.diag().expm1().log()
        self.net = net
        self.net.initialize_(torch.zeros_like(loc), self.cov_emb)
        self.reparameterized = reparameterized

    def model(self, trace, c, *shared):
        out = self.net(c[self.ext_from])
        mu, cov_emb = torch.split(out, [2, 2], dim=-1)
        cov = torch.diag_embed(F.softplus(cov_emb))

        value = trace.multivariate_normal(
            loc=mu,
            covariance_matrix=cov,
            name=self.ext_to,
            reparameterized=self.reparameterized,
        )
        return {self.ext_to: value}


class Tempered(Density):
    def __init__(
        self,
        name,
        d1: Union[Distribution, Density],
        d2: Union[Distribution, Density],
        beta: Tensor,
        optimize=False,
    ):
        assert torch.all(beta > 0.0) and torch.all(
            beta < 1.0
        ), "tempered densities are β=(0, 1) for clarity. Use model directly for β=0 or β=1"
        super().__init__(name, self.log_density_fn)
        self.optimize = optimize
        if self.optimize:
            self.logit = nn.Parameter(torch.logit(beta))
        else:
            self.logit = torch.logit(beta)
        self.density1 = d1
        self.density2 = d2

    def log_density_fn(self, value: Tensor) -> Tensor:
        def log_prob(g, value):
            return (
                g.log_density_fn(value)
                if isinstance(g, Density)
                else g.dist.log_prob(value)
            )

        return log_prob(self.density1, value) * (
            1 - torch.sigmoid(self.logit)
        ) + log_prob(self.density2, value) * torch.sigmoid(self.logit)

    def __repr__(self):
        return (
            "[β={:.4f}]".format(torch.sigmoid(self.logit).item()) + super().__repr__()
        )


class GMM(Density):
    def __init__(self, locs, covs, name="GMM"):
        assert len(locs) == len(covs)
        self.K = K = len(locs)
        super().__init__(name, self.log_density_fn)
        self.components = [
            distributions.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k])
            for k in range(K)
        ]

    def sample(self, sample_shape=torch.Size([1])):
        """only used to visualize samples"""
        zs = distributions.Categorical(
            torch.ones(len(self.components), device="cpu")
        ).sample(sample_shape=sample_shape)

        xs = []
        values, _ = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            x_k = self.components[k].sample(sample_shape=(n_k, *zs.shape[1:-1]))
            xs.append(x_k)

        xs = torch.cat(xs)
        ix = torch.randperm(xs.shape[0])

        return None, xs[ix].view(xs.size())

    def log_density_fn(self, value):
        lds = []
        for i, comp in enumerate(self.components):
            ld_i = comp.log_prob(value)
            lds.append(ld_i)
        lds_ = torch.stack(lds, dim=0)
        ld = torch.logsumexp(lds_, dim=0)
        return ld


class RingGMM(GMM):
    def __init__(self, name="RingGMM", radius=10, scale=0.5, count=8, device=None):
        alpha = 2 * math.pi / count
        x = radius * torch.sin(alpha * torch.arange(count).float())
        y = radius * torch.cos(alpha * torch.arange(count).float())
        locs = torch.stack((x, y), dim=0).T
        covs = torch.stack([torch.eye(2) * scale for m in range(count)], dim=0)
        super().__init__(name=name, locs=locs, covs=covs)


def anneal_between(left, right, total_num_targets, optimize_path=False):
    # Make an annealing path
    betas = torch.arange(0.0, 1.0, 1.0 / (total_num_targets - 1))[1:]  # g_0 is beta=0
    path = [
        Tempered(f"g{k}", left, right, beta, optimize=optimize_path)
        for k, beta in zip(range(1, total_num_targets - 1), betas)
    ]
    path = [left] + path + [right]
    assert len(path) == total_num_targets  # sanity check that the betas line up
    return path


def paper_model(num_targets=8, optimize_path=False):
    g0 = FixedMultivariateNormal(
        name=f"g0",
        loc=torch.zeros(2, **kw_autodevice()),
        cov=torch.eye(2, **kw_autodevice()) * 5 ** 2,
    )
    gK = RingGMM(radius=10, scale=0.5, count=8, name=f"g{num_targets - 1}").to(
        autodevice()
    )

    def paper_kernel(from_: int, to_: int, std: float, reparameterized: bool):
        return ConditionalMultivariateNormal(
            ext_from=f"g{from_}",
            ext_to=f"g{to_}",
            loc=torch.zeros(2, **kw_autodevice()),
            cov=torch.eye(2, **kw_autodevice()) * std ** 2,
            reparameterized=reparameterized,
            net=ResMLPJ(dim_in=2, dim_hidden=50, dim_out=2).to(autodevice()),
        )

    return dict(
        targets=anneal_between(g0, gK, num_targets, optimize_path=optimize_path),
        forwards=[
            paper_kernel(from_=i, to_=i + 1, std=1.0, reparameterized=True)
            for i in range(num_targets - 1)
        ],
        reverses=[
            paper_kernel(from_=i + 1, to_=i, std=1.0, reparameterized=False)
            for i in range(num_targets - 1)
        ],
    )
