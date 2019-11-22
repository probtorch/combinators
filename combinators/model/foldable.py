#!/usr/bin/env python3

from contextlib import contextmanager
import probtorch
import torch

from .. import graphs
from .model import Model
from ..sampler import Sampler

class Step(Model):
    def __init__(self, operator, initializer=None, iteration=0, ps=None,
                 walker=None, target_weights=None, **kwargs):
        assert isinstance(operator, Sampler)
        super(Step, self).__init__(batch_shape=operator.batch_shape)
        self._kwargs = kwargs
        self._iteration = iteration
        self._ps = ps
        self._walker = walker
        self._target_weights = target_weights

        self.add_module('operator', operator)
        if isinstance(initializer, Sampler):
            self.add_module('_initializer', initializer)
            assert self.operator.batch_shape == self._initializer.batch_shape
        else:
            self._initializer = initializer

    @property
    def name(self):
        return 'Step(%s)' % str(self._iteration)

    @contextmanager
    def _evaluate(self):
        if self._ps and self._ps.contains_model(self.name):
            operator = self.operator
            with self.operator.score(self._ps[self.name:]) as self.operator:
                if isinstance(self._initializer, Sampler):
                    initializer = self._initializer
                    with self._initializer.score(self._ps[self.name:]) as\
                         self._initializer:
                        yield self
                    self._initializer = initializer
                else:
                    yield self
            self.operator = operator
        else:
            yield self

    @contextmanager
    def _ready(self):
        with self._evaluate() as _:
            yield self

    def forward(self, *args, **kwargs):
        graph = graphs.ComputationGraph()
        with self._ready() as _:
            if isinstance(self._initializer, Sampler):
                seed, init_trace, seed_log_weight = self._initializer(**kwargs)
                graph.insert(self.name, init_trace)
            else:
                seed = self._initializer
                seed_log_weight = torch.zeros(self.batch_shape)
            result, op_trace, log_weight = self.operator(seed, *args, **kwargs)
            next_step = Step(self.operator, initializer=result,
                             iteration=self._iteration + 1, ps=self._ps,
                             walker=self._walker,
                             target_weights=self._target_weights,
                             **self._kwargs)
        if self._walker:
            next_step = self._walker(next_step)

        graph.insert(self.name, op_trace)
        log_weight = log_weight + seed_log_weight.to(log_weight)

        if isinstance(result, tuple):
            result = result + (next_step,)
        else:
            result = (result, next_step)

        return result, graph, log_weight

    def walk(self, f):
        if isinstance(self._initializer, Sampler):
            initializer = self._initializer.walk(f)
        else:
            initializer = self._initializer
        return f(Step(self.operator.walk(f), initializer, walker=f,
                      **self._kwargs))

    @contextmanager
    def score(self, ps):
        original_ps = self._ps
        try:
            self._ps = ps
            yield self
        finally:
            self._ps = original_ps

def step(operator, initializer=None, **kwargs):
    return Step(operator, initializer=initializer, **kwargs)

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
        graph = graphs.ComputationGraph()
        log_weight = torch.zeros(*self.batch_shape)
        moved_weight = False

        for item in items:
            step_results, step_trace, w = stepper(item, *args, **kwargs)
            graph.insert(self.name, step_trace)
            if not moved_weight:
                log_weight = log_weight.to(device=graph.device)
            log_weight += w
            stepper = step_results[-1]

        return step_results[:-1], graph, log_weight

    def walk(self, f):
        return f(Reduce(self.folder.walk(f), self._generator))

    @contextmanager
    def cond(self, qs):
        with self.folder.cond(qs[self.name:]) as folder_qs:
            yield self

def reduce(folder, generator):
    return Reduce(folder, generator)
