#!/usr/bin/env python3

import torch
from combinators.kernel import Kernel
from combinators.nnets import LinearMap

class NormalLinearKernel(Kernel):
    def __init__(self, ext_name):
        super().__init__()
        self.net = LinearMap(dim=1)
        self.ext_name = ext_name

    def apply_kernel(self, trace, cond_trace, cond_output):
        mu = self.net(cond_output.detach())
        return trace.normal(loc=mu, scale=torch.ones_like(mu), name=self.ext_name)

    def __repr__(self):
        return f'ext={self.ext_name}:' + super().__repr__()
