#!/usr/bin/env python3

from combinators.stochastic import ImproperRandomVariable, RandomVariable, Trace, Provenance
from combinators.program import Program, dispatch
from combinators.inference import Extend, Compose, Propose, Resample, copytraces, rerun_with_detached_values

# Optional
import combinators.trace.utils as trace_utils
from combinators.utils import adam, ppr, git_root, save_models, load_models
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.resamplers import Systematic

# Reexports
from torch import Tensor

_exports = \
  [Extend, Compose, Propose, Resample] + \
  [Program] + \
  [Trace, RandomVariable, ImproperRandomVariable, Provenance]

__all__ = [cls.__name__ for cls in _exports] + [
    "dispatch",
    "copytraces", "rerun_with_detached_values",

    "trace_utils",
    "adam", "ppr",
    "autodevice", "kw_autodevice",
    "effective_sample_size", "log_Z_hat",
    "Systematic",

    "Tensor"
]
