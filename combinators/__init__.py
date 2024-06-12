# Copyright 2021-2024 Northeastern University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3

from probtorch.stochastic import (
    ImproperRandomVariable,
    RandomVariable,
    Trace,
    Provenance,
)
from combinators.program import Program, dispatch
from combinators.inference import Extend, Compose, Propose, Resample
from combinators.out import global_store

# Optional
import combinators.trace.utils as trace_utils
from combinators.utils import adam, ppr, git_root, save_models, load_models
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.resamplers import Systematic

# Reexports
from torch import Tensor

_exports = (
    [Extend, Compose, Propose, Resample]
    + [Program]
    + [Trace, RandomVariable, ImproperRandomVariable, Provenance]
)

__all__ = [cls.__name__ for cls in _exports] + [
    "dispatch",
    "trace_utils",
    "adam",
    "ppr",
    "autodevice",
    "kw_autodevice",
    "effective_sample_size",
    "log_Z_hat",
    "Systematic",
    "global_store",
    "Tensor",
]
