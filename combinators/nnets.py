#!/usr/bin/env python

import torch
from torch import nn

class LinearMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        return self.net(x)

    def weight(self):
        return self.net.weight

    def bias(self):
        return self.net.bias

class ResMLPJ(nn.Module):
    """
    residual connection + MLP + joint layer
    """
    def __init__(self, dim_in, dim_hidden, dim_out, with_cov_embedding=False, initialize=None):
        assert initialize is None or initialize in ['truncated_normal']
        self._initialize_type = initialize
        super().__init__()
        self.with_cov_embedding = with_cov_embedding # TODO: ask heiko why this is ignored
        self.initialize = initialize
        self.joint = nn.Sequential(torch.nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = nn.Sequential(torch.nn.Linear(dim_hidden, dim_out))
        self.cov = nn.Sequential(torch.nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        y = self.joint(x)
        mu = self.mu(y) + x # Add residual connection because it is easier to move in relation to a thing than to learn from scratch
        cov_emb = self.cov(y)
        return mu, cov_emb

    def initialize_(self, loc_offset, cov_emb):
        if self._initialize_type is None or self._initialize_type in ['truncated_normal']:
            initer = nn.init.zeros_
        elif self._initialize_type == 'truncated_normal':
            initer = lambda aten: nn.init.normal_(aten, mean=0., std=0.01)
        else:
            raise TypeError()

        _ = [initer(ten) for ten in [self.cov[0].weight, self.cov[0].bias]]
        if self.with_cov_embedding:
            self.cov[0].bias.data.add_(cov_emb)
        else:
            self.mu[0].bias.data.add_(loc_offset)
