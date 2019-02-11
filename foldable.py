#!/usr/bin/env python3

import probtorch
import torch

import combinators
import graphs
import utils

class Foldable(combinators.Model):
    def __init__(self, operator, initializer=None, iteration=0, **kwargs):
        assert isinstance(operator, combinators.Sampler)
        super(Foldable, self).__init__(batch_shape=operator.batch_shape)
        self.add_module('operator', operator)
        if isinstance(initializer, combinators.Sampler):
            self.add_module('_initializer', initializer)
            assert self.operator.batch_shape == self._initializer.batch_shape
        else:
            self._initializer = initializer
        self._kwargs = kwargs
        self._iteration = iteration

    @property
    def name(self):
        return 'Foldable(%s)' % str(self._iteration)

    def forward(self, *args, **kwargs):
        graph = graphs.ModelGraph()
        if isinstance(self._initializer, combinators.Sampler):
            seed, init_trace, seed_weight = self._initializer(**kwargs)
            graph.insert(self.name, init_trace)
        else:
            seed = self._initializer
            seed_weight = torch.zeros(self.batch_shape)
        result, op_trace, weight = self.operator(seed, *args, **kwargs)
        next_step = Foldable(self.operator, initializer=result,
                             iteration=self._iteration + 1, **self._kwargs)

        graph.insert(self.name, op_trace)
        weight += seed_weight

        return (result, next_step), graph, weight

    def walk(self, f):
        if isinstance(self._initializer, combinators.Sampler):
            initializer = self._initializer.walk(f)
        else:
            initializer = self._initializer
        return f(Foldable(self.operator.walk(f), initializer, **self._kwargs))

    def cond(self, qs):
        qs_operator = qs[self.name:]
        if isinstance(self._initializer, combinators.Sampler):
            qs_initializer = qs[self.name:]
            initializer = self._initializer.cond(qs_initializer)
        else:
            initializer = self._initializer
        return Foldable(self.operator.cond(qs_operator), initializer,
                        **self._kwargs)

class Reduce(combinators.Model):
    def __init__(self, folder, generator):
        assert isinstance(folder, Foldable)
        super(Reduce, self).__init__(batch_shape=folder.batch_shape)
        self.add_module('folder', folder)
        self._generator = generator

    @property
    def name(self):
        return 'Reduce'

    def forward(self, *args, **kwargs):
        items = self._generator()
        stepper = self.folder
        trace = traces.Traces()
        weight = torch.zeros(self.batch_shape)

        for item in items:
            (step_result, next_step), step_trace, w = stepper(item, **kwargs)
            trace.insert(self.name, step_trace)
            weight += w
            stepper = next_step

        return step_result, trace, weight

    def walk(self, f):
        return f(Reduce(self.folder.walk(f), self._generator))

    def cond(self, qs):
        qs_folder = qs[self.name:]
        return Reduce(self.folder.cond(qs_folder), self._generator)
