#!/usr/bin/env python3

from combinators.stochastic import ImproperRandomVariable, RandomVariable, Trace
from combinators.program import Program, dispatch
from combinators.inference import Extend, Compose, Propose, Condition, Resample, copytraces, rerun_with_detached_values, maybe

# Optional
import combinators.trace.utils as trace_utils
from combinators.utils import adam, ppr, git_root, save_models, load_models
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.objectives import nvo_rkl, nvo_avo
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.resampling.strategies import Systematic

# Reexports
from torch import Tensor

_exports = \
  [Extend, Compose, Propose, Condition, Resample] + \
  [Program] + \
  [Trace, RandomVariable, ImproperRandomVariable]

__all__ = [cls.__name__ for cls in _exports] + [
    "dispatch",
    "copytraces", "rerun_with_detached_values", "maybe",

    "trace_utils",
    "adam", "ppr",
    "autodevice", "kw_autodevice",
    "nvo_rkl", "nvo_avo",
    "effective_sample_size", "log_Z_hat",
    "Systematic",

    "Tensor"
]
