#!/usr/bin/env python3

import torch
from torch.distributions import Normal

from combinators.inference import importance, mcmc
from combinators.kernel import kernel, mh
import combinators.model as model
import combinators.utils as utils

class AnnealingTarget(model.Primitive):
    def __init__(self, annealing_steps, *args, **kwargs):
        super(AnnealingTarget, self).__init__(*args, **kwargs)
        self._annealing_steps = annealing_steps

    @property
    def arguments(self):
        return (self._annealing_steps,)

    def _forward(self, t=0, data={}):
        beta = torch.linspace(0, 1, self._annealing_steps)[t]
        xs = self.sample(Normal, loc=torch.zeros(*self.batch_shape) * 3,
                         scale=torch.ones(*self.batch_shape) * 6, name='X_0')
        self.factor(self.p['X_0'].log_prob * (1 - beta), name='X_q')

        dist = Normal(loc=torch.zeros(*self.batch_shape),
                      scale=torch.ones(*self.batch_shape))
        self.factor(dist.log_prob(xs) * beta, name='X_p')

        return xs

def annealed_importance(target, transition, moves=1):
    return mcmc.Move(importance.importance(target), transition, moves=moves,
                     count_target=True)
