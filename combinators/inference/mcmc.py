#!/usr/bin/env python3

import torch

from .inference import Inference
from . import importance
from ..kernel.kernel import TransitionKernel
from ..kernel import mh
from ..model import foldable
from .. import utils

class Move(Inference):
    def __init__(self, target, kernel, moves=1, count_target=False):
        super(Move, self).__init__(target)
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
            log_weight = utils.normalize_weights(
                log_weight - importance.conditioning_factor({}, xi,
                                                            self.batch_shape)
            )
            xiq, log_weight_q = self.kernel(zs, xi, log_weight, *args, **kwargs)
            if not self._count_target:
                kwargs.pop('t')
            zs, xi, log_w = self.target.cond(xiq)(*args, **kwargs)
            log_omega_q = importance.conditioning_factor(xi, xiq,
                                                         self.batch_shape)
            log_weight = log_weight + utils.normalize_weights(
                log_weight_q - log_omega_q + log_w
            )

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

    def walk(self, f):
        return f(Move(self.target.walk(f), self.kernel, self._moves,
                      self._count_target))

    def cond(self, qs):
        return Move(self.target.cond(qs), self.kernel, self._moves,
                    self._count_target)

def move(target, kernel, moves=1):
    return Move(target, kernel, moves=moves)

def lightweight_mh(target, moves=1):
    return move(target, mh.LightweightKernel(target.batch_shape), moves=moves)

def resample_move_smc(target, kernel=mh.lightweight_kernel, moves=1):
    inference = lambda m: move(importance.resample(m), kernel(m.batch_shape),
                               moves)
    selector = lambda m: isinstance(m, importance.Propose)
    return target.apply(inference, selector)

def step_resample_move_smc(sampler, initializer=None, moves=1,
                           mcmc=lightweight_mh):
    return foldable.Step(mcmc(importance.resample(sampler), moves),
                         initializer=initializer)

def reduce_resample_move_smc(stepwise, step_generator, initializer=None,
                             moves=1, mcmc=lightweight_mh):
    rmsmc_foldable = step_resample_move_smc(stepwise, initializer=initializer,
                                            moves=moves, mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
