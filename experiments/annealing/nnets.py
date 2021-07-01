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
