#!/usr/bin/env python3

import torch

from .inference import Inference
from . import importance
from ..kernel.kernel import TransitionKernel
from ..kernel import mh
from ..model import foldable
from .. import utils

class MarkovChain(Inference):
    def __init__(self, target, kernel, moves=1):
        super(MarkovChain, self).__init__(target)
        assert isinstance(kernel, TransitionKernel)
        self.add_module('kernel', kernel)
        self._moves = moves

    def forward(self, *args, **kwargs):
        zs, xi, log_weight = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        for _ in range(self._moves):
            xiq = self.kernel(zs, xi, log_weight, *args, **kwargs)
            zsq, xiq, log_weight_q = self.target.cond(xiq)(*args, **kwargs)
            if self._accept(xi, log_weight, xiq, log_weight_q):
                zs = zsq
                xi = xiq
                log_weight = log_weight_q

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

class MHMove(MarkovChain):
    def _accept(self, xi, log_w, xiq, log_wq):
        marginal = utils.marginalize_all(log_w)
        marginal_q = utils.marginalize_all(log_wq)
        move_current = self.kernel.log_transition_prob(xiq, xi)
        move_candidate = self.kernel.log_transition_prob(xi, xiq)

        mh_ratio = (marginal_q - move_candidate) - (marginal - move_current)
        log_alpha = torch.min(torch.zeros(mh_ratio.shape), mh_ratio)
        return torch.bernoulli(torch.exp(log_alpha)) == 1

    def walk(self, f):
        return f(MHMove(self.target.walk(f), self.kernel, self._moves))

    def cond(self, qs):
        return MHMove(self.target.cond(qs), self.kernel, self._moves)

def mh_move(target, kernel, moves=1):
    return MHMove(target, kernel, moves=moves)

def lightweight_mh(target, moves=1):
    return mh_move(target, mh.LightweightKernel(target.batch_shape),
                   moves=moves)

def resample_move_smc(target, moves=1, mcmc=lightweight_mh):
    inference = lambda m: mcmc(importance.Resample(m), moves)
    selector = lambda m: isinstance(m, importance.Importance)
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
