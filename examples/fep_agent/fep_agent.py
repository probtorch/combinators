#!/usr/bin/env python3

import torch
from torch.distributions import Bernoulli, Dirichlet, Normal, RelaxedBernoulli
from torch.distributions import RelaxedOneHotCategorical
import torch.nn as nn

import combinators.model as model
import combinators.utils as utils

class FepAgent(model.Primitive):
    def __init__(self, *args, **kwargs):
        kwargs['params'] = {
            'meals': {
                'loc': torch.Tensor([3.]),
                'scale': torch.Tensor([1.]),
            },
            'foraging_success': {
                'temperature': torch.ones(24, 2),
                'probs': torch.Tensor([1e-6, 0.2]).expand(24, 2),
            },
            'survival_preference': {
                'temperature': torch.Tensor([1.]),
                'probs': torch.Tensor([1.]),
            },
        } if 'params' not in kwargs else kwargs['params']
        super(FepAgent, self).__init__(*args, **kwargs)
        self.policy = nn.Sequential(
            nn.Linear(1, 100),
            nn.Softsign(),
            nn.Linear(100, 24),
            nn.Sigmoid(),
        )

    def _forward(self, *args, **kwargs):
        success = self.param_sample(RelaxedBernoulli, name='foraging_success')
        preference = self.param_sample(RelaxedBernoulli,
                                       name='survival_preference')
        behavior = self.policy(preference)
        behavior, _ = utils.batch_collapse(behavior, self.batch_shape)
        success, _ = utils.batch_collapse(success, self.batch_shape)
        meals = torch.stack(
            [success[t, :, 0] * (1-behavior[t]) + success[t, :, 1] * behavior[t]
             for t in range(behavior.shape[0])],
            dim=0,
        ).sum(dim=1).unsqueeze(-1)
        return self.param_observe(Normal, name='meals', value=meals)
