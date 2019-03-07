#!/usr/bin/env python3

import numpy as np
import torch

from ..inference import importance
from .kernel import TransitionKernel
from .. import utils

class LightweightKernel(TransitionKernel):
    def log_transition_prob(self, origin, destination, shape):
        move = torch.zeros(*shape)
        move += destination.log_joint()
        move -= torch.log(torch.tensor(float(
            origin.num_variables(predicate=lambda k, v: not v.observed)
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
        return xiq, self.log_transition_prob(xi, xiq, log_weight.shape)

    def walk(self, f):
        return f(LightweightKernel(self.batch_shape))

    def cond(self, qs):
        return LightweightKernel(self.batch_shape)

    @property
    def name(self):
        return 'LightweightKernel'

def lightweight_kernel(shape):
    return LightweightKernel(shape)

class MHMoveKernel(TransitionKernel):
    def __init__(self, target, kernel, moves=1, *args, **kwargs):
        super(MHMoveKernel, self).__init__(
            *args, batch_shape=target.batch_shape, **kwargs
        )
        assert isinstance(kernel, TransitionKernel)
        self.add_module('kernel', kernel)
        self.add_module('target', target)
        self._moves = moves

    def log_transition_prob(self, origin, destination, shape):
        return self.kernel.log_transition_prob(origin, destination, shape)

    @property
    def name(self):
        return 'MHMoveKernel'

    def walk(self, f):
        return f(MHMoveKernel(self.target, self.kernel, self._moves))

    def cond(self, qs):
        return MHMoveKernel(self.target, self.kernel, self._moves,
                            q=qs[self.name])

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        for t in range(self._moves):
            xiq, log_weight_q = self.kernel(zs, xi, log_weight, *args,
                                            **kwargs)
            zsp, xip, log_w = self.target.cond(xiq)(*args, **kwargs)
            if not multiple_zs:
                zsp = (zsp,)
            log_omega_q = importance.conditioning_factor(xi, xiq,
                                                         self.batch_shape)
            log_w = log_weight_q - log_omega_q + log_w
            log_alpha = utils.batch_mean(
                torch.min(torch.zeros(self.batch_shape), log_w - log_weight)
            )
            if torch.bernoulli(log_alpha.exp()) == 1:
                zs = zsp
                xi = xip
                log_weight = log_w

        return xi, log_weight
