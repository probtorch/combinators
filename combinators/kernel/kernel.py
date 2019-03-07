#!/usr/bin/env python3

from probtorch.stochastic import RandomVariable
import torch

from combinators.sampler import Sampler
import combinators.utils as utils

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

class GaussianKernel(TransitionKernel):
    def __init__(self, model, var, scale=1.0):
        super(GaussianKernel, self).__init__()
        self._model = model
        self._scale = scale
        self._var = var

    def cond(self, qs):
        return GaussianKernel(self._model, self._var, self._scale)

    def walk(self, f):
        return f(self)

    @property
    def name(self):
        return 'GaussianKernel'

    def log_transition_prob(self, origin, destination):
        return destination[self._var].log_prob - origin[self._var].log_prob

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        q = utils.slice_trace(xi[self._model], self._var)
        var = xi[self._model][self._var]
        val = torch.normal(var.value, torch.ones(var.value.shape) * self._scale)
        q[self._var] = RandomVariable(var.dist, val, var.provenance, var.mask)
        xiq = xi.graft(self._model, q)
        return xiq, log_weight - var.log_prob + q[self._var].log_prob
