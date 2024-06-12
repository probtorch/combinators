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
from combinators import Propose, Extend, Compose, Resample
from experiments.annealing.objectives import stl_trace


def vsmc(targets, forwards, reverses, loss_fn, resample=False):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(
            p=Extend(p, rev),
            q=Compose(fwd, q, _debug=True),
            loss_fn=lambda out, loss: loss_fn(out) + loss,
            ix=k,
            _no_reruns=False,
            _debug=True,
            transf_q_trace=None if loss_fn.__name__ == "nvo_avo" else stl_trace,
        )
        if resample and k < len(forwards) - 1:
            q = Resample(q)
    return q
