#!/usr/bin/env python3

from combinators.sampler import Sampler

class TransitionKernel(Sampler):
    def __init__(self, batch_shape=(1,)):
        super(TransitionKernel, self).__init__()
        self._batch_shape = batch_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def forward(self, zs, xi, w, *args, **kwargs):
        raise NotImplementedError()

    def walk(self, f):
        raise NotImplementedError()

    def cond(self, qs):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def log_transition_prob(self, origin, destination):
        raise NotImplementedError()

    def get_model(self):
        return None
