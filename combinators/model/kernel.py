#!/usr/bin/env python3

import torch

from combinators.model.model import Model

class TransitionKernel(Model):
    def forward(self, zs, xi, w, *args, **kwargs):
        return zs, xi, w, torch.zeros(), torch.zeros()

    def walk(self, f):
        raise NotImplementedError()

    def cond(self, qs):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()
