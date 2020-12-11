#!/usr/bin/env python3
#
from torch.distributions import Distribution
from probtorch import Trace
from typing import Union, List
from abc import ABC


class Expr(object):
    pass


class LazyTrace(Expr):
    backend = Trace  # for mocking

    def __init__(self, *arg, **kwargs):
        """
        kwargs should be only tensors or distributions that have been observed
        under Trace machanisms, so we just use a trace here.
        """
        if len(arg) == 1 and isinstance(arg[0], Trace):
            self._trace = arg[0]
        elif len(arg) == 0:
            self._trace = self.backend()
        else:
            raise TypeError("LazyTrace only takes a positional trace")

        for k, v in kwargs.items():
            self._trace[k] = v

        self._keys = frozenset(self._trace.keys())  # not a frozenset because we like pain

    def __getattr__(self, x):
        return self._trace[x]

    def __getitem__(self, x):
        return self._trace[x]

    def keys(self):
        return self._keys

    def __str__(self):
        return "lazytrace[{}]".format(", ".join(["{}:{}".format(k, v) for k, v in self._trace.items()]))

    def __repr__(self):
        return self.__repr__()

    def __or__(self, other):
        return self

    def __and__(self, keys):
        """ find the subset of variables from a given collection of keys """
        return dict([(k, v) for k, v in self._trace.items() if k not in keys])

    def __add__(self, other):
        assert isinstance(other, self.__class__), "can only mappend {} instances".format(self.__class__)
        key_intersection = self.keys() & other.keys()
        if len(key_intersection) > 0:
            """ do this check both for the warning and because of kwarg collision """
            print("WARNING: overlapping keys found. mappend has right precidence.")
            # let the pain begin: this _should_ form a branch
            return LazyTrace(**(self & key_intersection), **other._trace)
        else:
            return LazyTrace(**self._trace, **other._trace)


lazytrace = LazyTrace
