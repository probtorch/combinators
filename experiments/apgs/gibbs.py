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
import torch.nn.functional as F
from combinators import Program, Compose, Propose, Resample, Extend
from .utils import apg_ix
from .objectives import loss_is, loss_apg


class Noop(Program):
    """We need this because Enc_coor is a kernel"""

    def __init__(self):
        super().__init__()

    def model(self, trace, c):
        return c


# Implement in a way that it extracts cov_kernel when t=0, and extracts z_where_{t-1} from c
def gibbs_sweeps(models, num_sweeps, T):
    q_enc_coor = models["enc_coor"]
    q_enc_digit = models["enc_digit"]
    p_dec_os = models["dec"]

    fwd_ix = lambda t, s: apg_ix(t, s, "forward")
    rev_ix = lambda t, s: apg_ix(t, s, "reverse")
    prp_ix = lambda t, s: apg_ix(t, s, "propose")
    _no_reruns = True

    q_os = Noop()
    for t in range(0, T):
        q_os = Compose(q1=q_os, q2=q_enc_coor, ix=fwd_ix(t, 0))
    q_os = Compose(q1=q_os, q2=q_enc_digit, ix=fwd_ix(T, 0))

    q_is = Propose(
        p=p_dec_os, q=q_os, ix=fwd_ix(T, 0), _no_reruns=_no_reruns, loss_fn=loss_is
    )

    if num_sweeps > 0:
        q_is = Resample(q_is, normalize_weights=True)

    q_t = q_is
    for sweep in range(1, num_sweeps + 1):  # Sweeps
        for t in range(T):  # Time step
            q_t = Propose(
                p=Extend(p_dec_os, q_enc_coor, ix=rev_ix(t, sweep)),
                q=Compose(q1=q_t, q2=q_enc_coor, _debug=True, ix=fwd_ix(t, sweep)),
                _no_reruns=_no_reruns,
                loss_fn=loss_apg,
                ix=prp_ix(t, sweep),
            )
            q_t = Resample(q_t, normalize_weights=True)
        q_t = Propose(
            p=Extend(p_dec_os, q_enc_digit, ix=rev_ix(T, sweep)),
            q=Compose(q1=q_t, q2=q_enc_digit, _debug=True, ix=fwd_ix(T, sweep)),
            _no_reruns=_no_reruns,
            loss_fn=loss_apg,
            ix=prp_ix(T, sweep),
        )
        if sweep != num_sweeps:
            q_t = Resample(q_t, normalize_weights=True)
    return q_t
