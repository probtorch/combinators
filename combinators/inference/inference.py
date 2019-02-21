#!/usr/bin/env python3

from probtorch.util import log_mean_exp

from combinators.sampler import Sampler

class Inference(Sampler):
    def __init__(self, target):
        super(Inference, self).__init__()
        assert isinstance(target, Sampler)
        self.add_module('target', target)

    @property
    def name(self):
        return self.get_model().name

    @property
    def batch_shape(self):
        return self.target.batch_shape

    def get_model(self):
        return self.target.get_model()

    def walk(self, f):
        raise NotImplementedError()

class Importance(Inference):
    def __init__(self, target, proposal):
        super(Importance, self).__init__(target)
        assert isinstance(proposal, Sampler)
        assert proposal.batch_shape == target.batch_shape
        self.add_module('proposal', proposal)

    def forward(self, *args, **kwargs):
        _, xi, w = self.proposal(*args, **kwargs)
        return self.target.cond(xi)(*args, **kwargs)

    def walk(self, f):
        return f(Importance(self.target.walk(f), self.proposal))

    def cond(self, qs):
        return Importance(self.target.cond(qs), self.proposal)

class Population(Inference):
    def __init__(self, target, batch_shape, before=True):
        super(Population, self).__init__(target)
        self._batch_shape = batch_shape
        self._before = before

    @property
    def batch_shape(self):
        return self._batch_shape + self.target.batch_shape

    @property
    def before(self):
        return self._before

    def forward(self, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        z, xi, w = self.target(*args, **kwargs)
        if not isinstance(z, tuple):
            z = (z,)
        if not self.before:
            z = self._expand_args(*z)
        return z, xi, w

    def walk(self, f):
        return f(Population(self.target.walk(f), batch_shape=self._batch_shape,
                            before=self.before))

    def cond(self, qs):
        return Population(self.target.cond(qs), batch_shape=self._batch_shape,
                          before=self.before)
