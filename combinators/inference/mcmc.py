#!/usr/bin/env python3

import torch

from .inference import Inference
from . import importance
from ..kernel.kernel import TransitionKernel
from ..kernel import mh
from ..model import foldable
from .. import utils

class MarkovChain(Inference):
    def __init__(self, target, kernel, moves=1, count_target=False):
        super(MarkovChain, self).__init__(target)
        assert isinstance(kernel, TransitionKernel)
        self.add_module('kernel', kernel)
        self._moves = moves
        self._count_target = count_target

    def forward(self, *args, **kwargs):
        zs, xi, log_weight = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        for t in range(self._moves):
            kwargs['t'] = t
            xiq, log_weight_q = self.kernel(zs, xi, log_weight, *args, **kwargs)
            if not self._count_target:
                kwargs.pop('t')
            zsp, xip, log_weight_p = self.target.cond(xiq)(*args, **kwargs)
            log_weight += log_weight_q + log_weight_p
            zs = zsp
            xi = xip

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

    def walk(self, f):
        return f(MarkovChain(self.target.walk(f), self.kernel, self._moves,
                             self._count_target))

    def cond(self, qs):
        return MarkovChain(self.target.cond(qs), self.kernel, self._moves,
                           self._count_target)

def mh_move(target, kernel, moves=1):
    return MarkovChain(target, kernel, moves=moves)

def lightweight_mh(target, moves=1):
    return mh_move(target, mh.LightweightKernel(target.batch_shape),
                   moves=moves)

def resample_move_smc(target, moves=1, mcmc=lightweight_mh):
    inference = lambda m: mcmc(importance.Resample(m), moves)
    selector = lambda m: isinstance(m, importance.Propose)
    return target.apply(inference, selector)

def step_resample_move_smc(sampler, initializer=None, moves=1,
                           mcmc=lightweight_mh):
    return foldable.Step(mcmc(importance.Resample(sampler), moves),
                         initializer=initializer)

def reduce_resample_move_smc(stepwise, step_generator, initializer=None,
                             moves=1, mcmc=lightweight_mh):
    rmsmc_foldable = step_resample_move_smc(stepwise, initializer=initializer,
                                            moves=moves, mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
