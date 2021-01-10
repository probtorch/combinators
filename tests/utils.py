#!/usr/bin/env python
from torch import Tensor
from collections import namedtuple
from pytest import mark, fixture
from typing import Optional, Callable
from combinators.debug import empirical_marginal_mean_std

Tolerance = namedtuple("Tolerance", ['loc', 'scale'])
Params = namedtuple("Params", ["loc", "scale"])

def assert_empirical_marginal_mean_std(runnable:Callable[[], Tensor], target_params:Params, tolerances:Tolerance, num_validate_samples = 400):
    eval_loc, eval_scale = empirical_marginal_mean_std(runnable, num_validate_samples = num_validate_samples)
    print("loc: {:.4f}, scale: {:.4f}".format(eval_loc, eval_scale))
    assert (target_params.loc  - tolerances.loc ) < eval_loc and  eval_loc < (target_params.loc  + tolerances.loc)
    assert (target_params.scale - tolerances.scale) < eval_scale and  eval_scale < (target_params.scale + tolerances.scale)

def g(i):
    return f"g{i}"
