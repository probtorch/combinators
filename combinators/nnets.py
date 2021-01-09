#!/usr/bin/env python
from torch import nn
from typing import *

class LinearMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        return self.net(x)

