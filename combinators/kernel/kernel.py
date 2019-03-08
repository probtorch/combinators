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
        log_transition = torch.zeros(self.batch_shape)
        origin_nodes = dict(origin.nodes())
        destination_nodes = dict(destination.nodes())
        for k, v in destination_nodes.items():
            fresh = k not in origin_nodes
            if isinstance(v, RandomVariable):
                fresh = fresh or (v.value != origin_nodes[k].value).any()
            else:
                fresh = fresh or (v.log_prob != origin_nodes[k].log_prob).any()
            if fresh:
                log_transition = log_transition.to(device=v.log_prob.device)
                log_transition = log_transition + v.log_prob
        return log_transition

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

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        q = utils.slice_trace(xi[self._model], self._var)
        var = xi[self._model][self._var]
        val = torch.normal(var.value, torch.ones(var.value.shape) * self._scale)
        q[self._var] = RandomVariable(var.dist, val, var.provenance, var.mask)
        xiq = xi.graft(self._model, q)
        return xiq, xiq.log_joint()

class LinScaledGaussianKernel(GaussianKernel):
    def __init__(self, model, var, scale=1.0, n_steps=1):
        super(LinScaledGaussianKernel, self).__init__(model, var, scale=scale)
        self._n_steps = n_steps
        self._init_scale = scale

    def cond(self, qs):
        return LinScaledGaussianKernel(self._model, self._var, self._scale, self._maxT)

    def walk(self, f):
        return f(self)

    @property
    def name(self):
        return 'LinScaledGaussianKernel'

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        self._scale = self._init_scale*max([1-(kwargs['t']/self._n_steps), 0.1])
        return super(LinScaledGaussianKernel, self).forward(zs, xi, log_weight, *args, **kwargs)
