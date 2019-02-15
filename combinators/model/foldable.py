#!/usr/bin/env python3

import probtorch
import torch

from .. import graphs
from .model import Model
from ..sampler import Sampler
from .. import utils

class Step(Model):
    def __init__(self, operator, initializer=None, iteration=0, qs=None,
                 **kwargs):
        assert isinstance(operator, Sampler)
        super(Step, self).__init__(batch_shape=operator.batch_shape)
        self._kwargs = kwargs
        self._iteration = iteration
        self._qs = qs

        if self._qs and self._qs.contains_model(self.name):
            qs = self._qs[self.name:]
            operator = operator.cond(qs)
            if isinstance(initializer, Sampler):
                initializer = initializer.cond(qs)

        self.add_module('operator', operator)
        if isinstance(initializer, Sampler):
            self.add_module('_initializer', initializer)
            assert self.operator.batch_shape == self._initializer.batch_shape
        else:
            self._initializer = initializer

    @property
    def name(self):
        return 'Step(%s)' % str(self._iteration)

    def forward(self, *args, **kwargs):
        graph = graphs.ModelGraph()
        if isinstance(self._initializer, Sampler):
            seed, init_trace, seed_log_weight = self._initializer(**kwargs)
            graph.insert(self.name, init_trace)
        else:
            seed = self._initializer
            seed_log_weight = torch.zeros(self.batch_shape)
        result, op_trace, log_weight = self.operator(seed, *args, **kwargs)
        next_step = Step(self.operator, initializer=result,
                         iteration=self._iteration + 1, qs=self._qs,
                         **self._kwargs)

        graph.insert(self.name, op_trace)
        log_weight += seed_log_weight

        return (result, next_step), graph, log_weight

    def walk(self, f):
        if isinstance(self._initializer, Sampler):
            initializer = self._initializer.walk(f)
        else:
            initializer = self._initializer
        return Step(self.operator.walk(f), initializer, **self._kwargs)

    def cond(self, qs):
        return Step(self.operator, self._initializer, self._iteration, qs,
                    **self._kwargs)

def step(operator, initializer=None, qs=None, **kwargs):
    return Step(operator, initializer=initializer, qs=qs, **kwargs)

class Reduce(Model):
    def __init__(self, folder, generator):
        assert isinstance(folder.get_model(), Step)
        super(Reduce, self).__init__(batch_shape=folder.batch_shape)
        self.add_module('folder', folder)
        self._generator = generator

    @property
    def name(self):
        return 'Reduce'

    def forward(self, *args, **kwargs):
        items = self._generator()
        stepper = self.folder
        graph = graphs.ModelGraph()
        log_weight = torch.zeros(self.batch_shape)

        for item in items:
            (step_result, next_step), step_trace, w = stepper(item, *args,
                                                              **kwargs)
            graph.insert(self.name, step_trace)
            log_weight += w
            stepper = next_step

        return step_result, graph, log_weight

    def walk(self, f):
        return Reduce(self.folder.walk(f), self._generator)

    def cond(self, qs):
        qs_folder = qs[self.name:]
        return Reduce(self.folder.cond(qs_folder), self._generator)

def reduce(folder, generator):
    return Reduce(folder, generator)
