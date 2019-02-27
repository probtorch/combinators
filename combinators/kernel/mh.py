#!/usr/bin/env python3

import numpy as np
import torch

from .kernel import TransitionKernel
from .. import utils

class LightweightKernel(TransitionKernel):
    def log_transition_prob(self, source, destination):
        move = utils.marginalize_all(destination.log_joint())
        move -= torch.log(torch.tensor(float(
            source.num_variables(predicate=lambda k, v: not v.observed)
        )))
        return move

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        sampled = []
        while not sampled:
            t = np.random.randint(len(xi))
            sampled = [k for k in xi[t].variables() if not xi[t][k].observed]

        key = sampled[np.random.randint(len(sampled))]
        candidate_trace = utils.slice_trace(xi[t], key)

        xiq = xi.graft(t, candidate_trace)
        return xiq, self.log_transition_prob(xi, xiq)

    def walk(self, f):
        return f(LightweightKernel(self.batch_shape))

    def cond(self, qs):
        return LightweightKernel(self.batch_shape)

    @property
    def name(self):
        return 'LightweightKernel'

def lightweight_kernel(shape):
    return LightweightKernel(shape)
