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
import torch
import torch.distributions as D
from combinators.resamplers import Systematic
from probtorch.stochastic import Trace, RandomVariable


def test_ancestor_indices_systematic():
    S = 4
    B = 1000
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
    lw = lw.unsqueeze(1).expand(S, B)
    a = Systematic().ancestor_indices_systematic(lw, 0, 1).T
    for i in range(S):
        print(i, (a == i).sum() / (S * B))


def test_resample_with_batch(B=100, N=5):
    S = 4

    value = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()

    lw = lw.unsqueeze(1).expand(S, B)
    value = value.unsqueeze(1).expand(S, B, 2)
    tr = Trace()

    for n in range(N):
        tr._inject(
            RandomVariable(
                dist=D.Normal(0, 1), value=value, log_prob=lw, reparameterized=False
            ),
            name=f"z_{n}",
            silent=True,
        )

    resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=1)
    assert (_lw.exp() == 0.25).all()

    memo = torch.zeros(S)
    for n, (_, rv) in enumerate(resampled.items()):
        for s in range(S):
            memo[s] += (rv.value == (s + 1)).sum() / (S * B * N * 2)

    print(memo)


def test_resample_without_batch():
    S = 4
    N = 5
    B = 100

    value = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
    tr = Trace()

    memo = torch.zeros(S)
    for _ in range(B):
        for n in range(N):
            tr._inject(
                RandomVariable(
                    dist=D.Normal(0, 1), value=value, log_prob=lw, reparameterized=False
                ),
                name=f"z_{n}",
            )

        resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=None)

        assert (_lw.exp() == 0.25).all()
        for n, (_, rv) in enumerate(resampled.items()):
            for s in range(S):
                memo[s] += (rv.value == (s + 1)).sum() / (S * N * 2)

    print(memo / B)
