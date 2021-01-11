#!/usr/bin/env python3

import torch
from combinators.kernel import Kernel
from combinators.nnets import LinearMap


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
                            scale=torch.ones_like(mu),
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
