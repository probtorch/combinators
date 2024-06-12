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
#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import Tensor


def effective_sample_size(lw: Tensor, sample_dims=-1) -> Tensor:
    lnw = F.softmax(lw, dim=sample_dims).log()
    ess = torch.exp(-torch.logsumexp(2 * lnw, dim=sample_dims))
    return ess


def log_Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return Z_hat(lw, sample_dims=sample_dims).log()


def Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return lw.exp().mean(dim=sample_dims)
