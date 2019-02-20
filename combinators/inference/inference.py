#!/usr/bin/env python3

from combinators.sampler import Sampler

class Inference(Sampler):
    def __init__(self, sampler):
        super(Inference, self).__init__()
        assert isinstance(sampler, Sampler)
        self.add_module('sampler', sampler)

    @property
    def name(self):
        return self.get_model().name

    @property
    def batch_shape(self):
        return self.sampler.batch_shape

    def get_model(self):
        return self.sampler.get_model()

    def walk(self, f):
        raise NotImplementedError()

class GuidedConditioning(Inference):
    def __init__(self, sampler, guide):
        super(GuidedConditioning, self).__init__(sampler)
        assert isinstance(guide, Sampler)
        assert guide.batch_shape == sampler.batch_shape
        self.add_module('guide', guide)

    def forward(self, *args, **kwargs):
        _, xi, w = self.guide(*args, **kwargs)
        return self.sampler.cond(xi)(*args, **kwargs)

    def walk(self, f):
        return f(GuidedConditioning(self.sampler.walk(f), self.guide))

    def cond(self, qs):
        return GuidedConditioning(self.sampler.cond(qs), self.guide)

class Population(Inference):
    def __init__(self, sampler, batch_shape, before=True):
        super(Population, self).__init__(sampler)
        self._batch_shape = batch_shape
        self._before = before

    @property
    def batch_shape(self):
        return self._particle_shape + self.sampler.batch_shape

    @property
    def before(self):
        return self._before

    def forward(self, *args, **kwargs):
        if self.before:
            args, kwargs = self._expand_args(*args, **kwargs)
        z, xi, w = self.sampler(*args, **kwargs)
        if not isinstance(z, tuple):
            z = (z,)
        if not self.before:
            z = self._expand_args(*z)
        return z, xi, w

    def walk(self, f):
        return f(Population(self.sampler.walk(f), batch_shape=self._batch_shape,
                            before=self.before))

    def cond(self, qs):
        return Population(self.sampler.cond(qs), batch_shape=self._batch_shape,
                          before=self.before)
