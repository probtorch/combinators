#!/usr/bin/env python
import torch
from torch import Tensor
from collections import namedtuple
from pytest import mark, fixture
from typing import Optional, Callable
from combinators.debug import empirical_marginal_mean_std
import os
import numpy as np

Tolerance = namedtuple("Tolerance", ['loc', 'scale'])
Params = namedtuple("Params", ["loc", "scale"])

def assert_empirical_marginal_mean_std(runnable:Callable[[], Tensor], target_params:Params, tolerances:Tolerance, is_smoketest_bool=False, num_validate_samples = 400):
    eval_loc, eval_scale = empirical_marginal_mean_std(runnable, num_validate_samples = num_validate_samples)
    print("loc: {:.4f}, scale: {:.4f}".format(eval_loc, eval_scale))
    assert is_smoketest_bool or abs(target_params.loc - eval_loc) < tolerances.loc, f'{eval_loc} is not within +-{tolerances.loc} of {target_params.loc}'
    assert is_smoketest_bool or abs(target_params.scale - eval_scale) < tolerances.scale, f'{eval_scale} is not within +-{tolerances.scale} of {target_params.scale}'

def g(i):
    return f"g{i}"

@fixture
def is_smoketest():
    env_var = os.getenv('COMBINATORS_SMOKETEST') is not None
    return env_var

@fixture(autouse=True)
def seed():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.set_deterministic(True)
