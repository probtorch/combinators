#!/usr/bin/env python3
import torch
from combinators.stochastic import Trace
from typing import Callable, Any, Tuple, Optional, Set
from copy import deepcopy
from typeguard import typechecked


@typechecked
def remains_pure(tr: Trace) -> Callable[[Trace], bool]:
    """ TODO """
    cached = tr.detach().deepcopy()  # something like this to detach and copy all tensors

    @typechecked
    def check(tr: Trace) -> None:
        if cached is not None:
            # ensure check that everything is the same
            del cached
            return True
        else:
            raise RuntimeError("You can only perform this check once per operation")

    return check



# FIXME: currently not used, but currying the annotations might be nice
def curry(func):
    """ taken from: https://www.python-course.eu/currying_in_python.php """
    curry.__curried_func_name__ = func.__name__
    f_args, f_kwargs = [], {}
    def f(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            return f
        else:
            result = func(*f_args, *f_kwargs)
            f_args, f_kwargs = [], {}
            return result
    return f
