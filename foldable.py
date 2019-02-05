#!/usr/bin/env python3

import probtorch
import torch

import combinators
import traces
import utils

class Foldable(combinators.ModelSampler):
    def __init__(self, operator, initializer=None, **kwargs):
        super(Foldable, self).__init__()
        assert isinstance(operator, combinators.Sampler)
        self.add_module('operator', operator)
        if isinstance(initializer, combinators.Sampler):
            self.add_module('_initializer', initializer)
        else:
            self._initializer = initializer
        self._kwargs = kwargs

    @property
    def name(self):
        return 'Foldable(%s)' % self.operator.name

    def _forward(self, *args, **kwargs):
        if isinstance(self._initializer, combinators.Sampler):
            seed, trace = self._initializer(**kwargs)
        else:
            seed = self._initializer
            trace = kwargs.pop('trace')
        kwargs['trace'] = trace.extract(self.name + str(args))
        result, op_trace = self.operator(seed, *args, **kwargs)
        trace.insert(self.name + str(args), op_trace)
        next_step = Foldable(self.operator, result, **self._kwargs)
        return (result, next_step), trace

class Reduce(combinators.ModelSampler):
    def __init__(self, folder, generator):
        super(Reduce, self).__init__()
        assert isinstance(folder, Foldable)
        self.add_module('folder', folder)
        self._generator = generator

    @property
    def name(self):
        return 'Reduce(%s)' % self.folder.name

    def _forward(self, *args, **kwargs):
        trace = kwargs.pop('trace')
        items = self._generator()
        stepper = self.folder

        for item in items:
            kwargs['trace'] = trace.extract(self.name + '/' + str(item))
            (step_result, stepper), step_trace = stepper(item, **kwargs)
            trace.insert(self.name + '/' + str(item), step_trace)

        return step_result, trace
