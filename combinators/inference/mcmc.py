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
            log_weight -= importance.conditioning_factor(
                {}, xi, self.batch_shape
            )
            xiq, _ = self.kernel(zs, xi, log_weight, *args, **kwargs)
            if not self._count_target:
                kwargs.pop('t')
            zs, xi, log_weight = importance.conditioned_evaluate(
                self.target, xiq, *args, **kwargs
            )
            if not multiple_zs:
                zs = (zs,)

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

    def walk(self, f):
        return f(Move(self.target.walk(f), self.kernel, self._moves,
                      self._count_target))

    def cond(self, qs):
        return Move(self.target.cond(qs), self.kernel, self._moves,
                    self._count_target)

class MetropolisHastings(Inference):
    def __init__(self, target, kernel, moves=1, count_target=False):
        super(MetropolisHastings, self).__init__(target)
        assert isinstance(kernel, TransitionKernel)
        self.add_module('kernel', kernel)
        self._moves = moves
        self._count_target = count_target

    def walk(self, f):
        return f(MetropolisHastings(self.target.walk(f), self.kernel,
                                    self._moves, self._count_target))

    def cond(self, qs):
        return MetropolisHastings(self.target.cond(qs), self.kernel,
                                  self._moves, self._count_target)

    def forward(self, *args, **kwargs):
        zs, xi, log_weight = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        for t in range(self._moves):
            # Rescore the current trace under any change of target based on t.
            _, xi, log_weight = self.target.rescore(xi)(*args, **kwargs)
            kwargs['t'] = t
            xiq, log_weight_q = self.kernel(zs, xi, log_weight, *args,
                                            **kwargs)
            if not self._count_target:
                kwargs.pop('t')
            zsp, xip, log_w = importance.conditioned_evaluate(self.target, xiq,
                                                              *args, **kwargs)
            if not multiple_zs:
                zsp = (zsp,)
            log_transition = self.kernel.log_transition_prob(xi, xip)
            log_reverse_transition = self.kernel.log_transition_prob(xip, xi)
            log_w = log_weight_q + log_w
            log_alpha = utils.batch_marginalize(torch.min(
                torch.zeros(self.batch_shape),
                (log_w + log_reverse_transition) - (log_weight + log_transition)
            ))
            if utils.is_number(log_alpha) and\
               (torch.bernoulli(log_alpha.exp()) == 1).all():
                zs = zsp
                xi = xip
                log_weight = log_w

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

def move(target, kernel, moves=1):
    return Move(target, kernel, moves=moves)

def lightweight_mh(target, moves=1):
    return move(target, mh.LightweightKernel(target.batch_shape), moves=moves)

def resample_move_smc(target, kernel=mh.lightweight_kernel, moves=1):
    inference = lambda m: move(importance.resample(m), kernel(m.batch_shape),
                               moves)
    selector = lambda m: isinstance(m, importance.Propose)
    return target.apply(inference, selector)

def step_resample_move_smc(target, kernel=mh.lightweight_kernel, moves=1):
    inference = lambda m: move(importance.resample(m), kernel(m.batch_shape),
                               moves)
    selector = lambda m: isinstance(m, foldable.Step)
    return target.apply(inference, selector)
