#!/usr/bin/env python3

import torch
from torch.distributions import Normal

import combinators.model as model
import combinators.utils as utils

class AnnealingProposal(model.Primitive):
    def _forward(self, *args, data={}):
        return self.sample(Normal, loc=torch.zeros(*self.batch_shape),
                           scale=torch.ones(*self.batch_shape), name='X_0')

class AnnealingTarget(model.Primitive):
    def __init__(self, annealing_steps, *args, **kwargs):
        self._annealing_steps = annealing_steps
        super(AnnealingTarget, self).__init__(*args, **kwargs)
        self.proposal = AnnealingProposal(batch_shape=self.batch_shape)

    @property
    def arguments(self):
        return (self._annealing_steps,)

    def _forward(self, t=0, data={}):
        beta = torch.linspace(0, 1, self._annealing_steps)[t]
        xs, p, _ = self.proposal()
        self.p = utils.join_traces(self.p, p['AnnealingProposal'])
        self.factor(self.p['X_0'].log_prob * (1 - beta), name='X_q')

        dist = Normal(loc=torch.ones(*self.batch_shape) * 3,
                      scale=torch.ones(*self.batch_shape) * 6)
        self.factor(dist.log_prob(xs) * beta, name='X_p')

        return xs
