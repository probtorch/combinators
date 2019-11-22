#!/usr/bin/env python3

import numpy as np
import torch

from ..inference import importance
from .kernel import TransitionKernel
from .. import utils

class LightweightKernel(TransitionKernel):
    def forward(self, zs, xi, *args, **kwargs):
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

    def score(self, ps):
        return LightweightKernel(self.batch_shape)

    @property
    def name(self):
        return 'LightweightKernel'

def lightweight_kernel(shape):
    return LightweightKernel(shape)
