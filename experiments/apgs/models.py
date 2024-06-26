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
import math
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from probtorch.stochastic import Provenance, RandomVariable
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from combinators import Program

from experiments.apgs.affine_transformer import Affine_Transformer
from .utils import global_store, key, is_forward


def init_models(
    frame_pixels,
    shape_pixels,
    num_hidden_digit,
    num_hidden_coor,
    z_where_dim,
    z_what_dim,
    num_objects,
    mean_shape,
    device,
    reparameterized=False,
):
    AT = Affine_Transformer(frame_pixels, shape_pixels, device)

    models = dict()

    models["dec"] = DecoderFull(
        num_pixels=shape_pixels ** 2,
        num_hidden=num_hidden_digit,
        z_where_dim=z_where_dim,
        z_what_dim=z_what_dim,
        AT=AT,
        device=device,
        reparameterized=reparameterized,
        num_objects=num_objects,
    ).to(device)

    models["enc_coor"] = Enc_coor(
        num_pixels=(frame_pixels - shape_pixels + 1) ** 2,
        mean_shape=mean_shape,
        num_hidden=num_hidden_coor,
        z_where_dim=z_where_dim,
        AT=AT,
        dec=models["dec"],
        num_objects=num_objects,
        reparameterized=reparameterized,
    ).to(device)

    models["enc_digit"] = Enc_digit(
        num_pixels=shape_pixels ** 2,
        num_hidden=num_hidden_digit,
        z_what_dim=z_what_dim,
        AT=AT,
        reparameterized=reparameterized,
    ).to(device)

    return models


class Enc_coor(Program):
    """
    encoder of the digit positions
    """

    def __init__(
        self,
        num_pixels,
        num_hidden,
        z_where_dim,
        AT,
        dec,
        mean_shape,
        num_objects,
        reparameterized=False,
    ):
        super().__init__()
        self.enc_coor_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden), nn.ReLU()
        )
        self.where_mean = nn.Sequential(
            nn.Linear(num_hidden, int(0.5 * num_hidden)),
            nn.ReLU(),
            nn.Linear(int(0.5 * num_hidden), z_where_dim),
            nn.Tanh(),
        )

        self.where_log_std = nn.Sequential(
            nn.Linear(num_hidden, int(0.5 * num_hidden)),
            nn.ReLU(),
            nn.Linear(int(0.5 * num_hidden), z_where_dim),
        )
        self.AT = AT
        self.get_dec = lambda: dec
        self.mean_shape = mean_shape
        self.K = num_objects
        self.reparameterized = reparameterized

    def model(self, trace, c, ix):
        frames = c["frames"]
        S, B, T, FP, _ = frames.shape

        if ix.sweep == 0:
            # FIXME: Figure out if we can use cheaper expand here
            conv_kernel = self.mean_shape.repeat(S, B, self.K, 1, 1)
        else:
            z_what_val = c[key.z_what(ix.sweep - 1)]
            conv_kernel = self.get_dec().get_conv_kernel(z_what_val)

        _, _, K, DP, _ = conv_kernel.shape
        frame_left = frames[:, :, ix.t, :, :]

        q_mean, q_std = (
            torch.zeros(S, B, self.K, 2).to(frames.device),
            torch.zeros(S, B, self.K, 2).to(frames.device),
        )
        z_where_t = torch.zeros(S, B, self.K, 2).to(frames.device)

        # K objects in frame
        for k in range(self.K):
            conved_k = F.conv2d(
                frame_left.view(S * B, FP, FP).unsqueeze(0),
                conv_kernel[:, :, k, :, :].view(S * B, DP, DP).unsqueeze(1),
                groups=int(S * B),
            )
            CP = conved_k.shape[-1]  # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(
                conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP * CP), -1
            )  ## S * B * 1639
            hidden = self.enc_coor_hidden(conved_k)
            q_mean_k = self.where_mean(hidden)
            q_std_k = self.where_log_std(hidden).exp()
            q_mean[:, :, k, :] = q_mean_k
            q_std[:, :, k, :] = q_std_k

            if is_forward(ix):
                # bulid up z_where
                z_where_val_k = Normal(q_mean_k, q_std_k).sample()
                z_where_t[:, :, k, :] = z_where_val_k
            else:
                # retreive z_where
                z_where_val_k = trace._cond_trace[
                    key.z_where(ix.t, ix.sweep - 1)
                ].value[:, :, k, :]

            recon_k = (
                self.AT.digit_to_frame(
                    conv_kernel[:, :, k, :, :].unsqueeze(2),
                    z_where_val_k.unsqueeze(2).unsqueeze(2),
                )
                .squeeze(2)
                .squeeze(2)
            )
            assert recon_k.shape == (S, B, FP, FP), "shape = %s" % recon_k.shape
            frame_left = frame_left - recon_k

        # Stuff happens in a Compose
        if is_forward(ix):
            # For performace reasons we want to add all K objects as one RV, hence we need to cheat here:
            # We sampled all K RVs manually in for-loop above, and "simulate" a combinators sampling operation here.
            # if ix.t <= 1 and ix.sweep > 0:
            trace.normal(
                loc=q_mean,
                scale=q_std,
                value=z_where_t,
                provenance=Provenance.SAMPLED,
                reparameterized=self.reparameterized,
                name=key.z_where(ix.t, ix.sweep),
            )

            global_store[key.z_where(ix.t, ix.sweep)] = weakref.ref(z_where_t)

            # We need this because in initial IS step this is not run as a "kernel"
            if ix.sweep == 0:
                return {**c, key.z_where(ix.t, ix.sweep): z_where_t}
            return c
        else:
            trace.normal(
                loc=q_mean,
                scale=q_std,
                name=key.z_where(ix.t, ix.sweep - 1),
                reparameterized=self.reparameterized,
            )
            return None


class Enc_digit(Program):
    """
    encoder of digit features
    """

    def __init__(self, num_pixels, num_hidden, z_what_dim, AT, reparameterized=False):
        super().__init__()
        self.enc_digit_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, int(0.5 * num_hidden)),
            nn.ReLU(),
        )
        self.enc_digit_mean = nn.Sequential(
            nn.Linear(int(0.5 * num_hidden), z_what_dim)
        )
        self.enc_digit_log_std = nn.Sequential(
            nn.Linear(int(0.5 * num_hidden), z_what_dim)
        )
        self.AT = AT
        self.reparameterized = reparameterized

    def model(self, trace, c, ix):
        frames = c["frames"]
        # z_where are fetched from the input in the for loop below

        sample_shape = frames.shape[:3]
        data_shape = c[key.z_where(0, ix.sweep)].shape[-2:]
        z_where = torch.zeros(*sample_shape, *data_shape, device=frames.device)

        for t in range(frames.shape[2]):
            z_where[:, :, t, :, :] = c[key.z_where(t, ix.sweep)]

        cropped = self.AT.frame_to_digit(frames=frames, z_where=z_where)
        cropped = torch.flatten(cropped, -2, -1)
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden)
        q_std = self.enc_digit_log_std(hidden).exp()

        sweep_ix = ix.sweep if is_forward(ix) else (ix.sweep - 1)
        z_what = trace.normal(
            loc=q_mu,
            scale=q_std,
            name=key.z_what(sweep_ix),
            reparameterized=self.reparameterized,
        )
        if is_forward(ix):
            global_store[key.z_what(sweep_ix)] = weakref.ref(z_what)

        return None


class _Decoder(Program):
    def __init__(
        self,
        num_pixels,
        num_objects,
        num_hidden,
        z_where_dim,
        z_what_dim,
        AT,
        device,
        reparameterized=False,
    ):
        super().__init__()
        self.dec_digit_mean = nn.Sequential(
            nn.Linear(z_what_dim, int(0.5 * num_hidden)),
            nn.ReLU(),
            nn.Linear(int(0.5 * num_hidden), num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_pixels),
            nn.Sigmoid(),
        )

        self.prior_where0_mu = torch.zeros(z_where_dim, device=device)
        self.prior_where0_Sigma = torch.ones(z_where_dim, device=device) * 1.0
        self.prior_wheret_Sigma = torch.ones(z_where_dim, device=device) * 0.2
        self.prior_what_mu = torch.zeros(z_what_dim, device=device)
        self.prior_what_std = torch.ones(z_what_dim, device=device)
        self.AT = AT
        self.K = num_objects
        self.reparameterized = reparameterized

    # In q_\phi case
    def get_conv_kernel(self, z_what_value, detach=True):
        digit_mean = self.dec_digit_mean(z_what_value)  # S * B * K * (28*28)
        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        digit_mean = digit_mean.detach() if detach else digit_mean
        return digit_mean

    def get_z_where(self, trace, T, ix, sample_shape):
        ##################################################################
        # Remove the optimization by always computing the log joint.
        # Compute the log prior of z_where_{1:T} and construct a sample
        # trajectory using sample values
        # 1. from sweep for preceding and current timesteps i.e. t <= ix.t
        # 2. from sweep-1 for future timesteps, i.e. t > ix.t
        ##################################################################
        data_shape = (self.K, *self.prior_where0_mu.shape)
        z_where_vals = torch.zeros(
            *sample_shape, *data_shape, device=self.prior_where0_mu.device
        )

        for t in range(T):
            if t <= ix.t:
                if t == 0:
                    trace.normal(
                        loc=self.prior_where0_mu,
                        scale=self.prior_where0_Sigma,
                        name=key.z_where(0, ix.sweep),
                        reparameterized=self.reparameterized,
                    )
                else:
                    trace.normal(
                        loc=global_store[key.z_where(t - 1, ix.sweep)](),
                        scale=self.prior_wheret_Sigma,
                        name=key.z_where(t, ix.sweep),
                        reparameterized=self.reparameterized,
                    )
                z_where_vals[:, :, t, :, :] = trace[key.z_where(t, ix.sweep)].value

            elif t == (ix.t + 1):
                trace.normal(
                    loc=global_store[key.z_where(ix.t, ix.sweep)](),
                    scale=self.prior_wheret_Sigma,
                    name=key.z_where(t, ix.sweep - 1),
                    reparameterized=self.reparameterized,
                )
                z_where_vals[:, :, t, :, :] = trace[key.z_where(t, ix.sweep - 1)].value

            else:
                trace.normal(
                    loc=global_store[key.z_where(ix.t, ix.sweep)](),
                    scale=self.prior_wheret_Sigma,
                    name=key.z_where(t, ix.sweep - 1),
                    reparameterized=self.reparameterized,
                )
                z_where_vals[:, :, t, :, :] = trace[key.z_where(t, ix.sweep - 1)].value
        return z_where_vals

    def get_z_what_val(self, trace, T, ix):
        # index for z_what given ix
        if ix.t == T:
            z_what_index = ix.sweep
        elif ix.t < T and ix.sweep > 0:
            z_what_index = ix.sweep - 1
        else:
            raise ValueError(
                f"You should not call the decoder when t={ix.t} and sweep={ix.sweep}"
            )

        z_what_val = trace.normal(
            loc=self.prior_what_mu,
            scale=self.prior_what_std,
            name=key.z_what(z_what_index),
            reparameterized=self.reparameterized,
        )
        return z_what_index, z_what_val


class DecoderFull(_Decoder):
    def model(self, trace, c, ix, EPS=1e-9):
        frames = c["frames"]
        _, _, T, FP, _ = frames.shape
        ##################################################################
        # Remove the optimization by always computing the log joint.
        # Compute the log prior of z_where_{1:T} and construct a sample
        # trajectory using sample values
        # 1. from sweep for preceding and current timesteps i.e. t <= ix.t
        # 2. from sweep-1 for future timesteps, i.e. t > ix.t
        ##################################################################
        z_where_vals = self.get_z_where(trace, T, ix, sample_shape=frames.shape[:3])
        z_what_index, z_what_val = self.get_z_what_val(trace, T, ix)

        digit_mean = self.get_conv_kernel(z_what_val, detach=False)
        recon_frames = torch.clamp(
            self.AT.digit_to_frame(digit_mean, z_where_vals).sum(-3), min=0.0, max=1.0
        )
        _ = trace.variable(
            Bernoulli,
            probs=recon_frames + EPS,
            value=frames,
            name="recon",
            provenance=Provenance.OBSERVED,
            reparameterized=self.reparameterized,
        )

        z_wheres = {
            key.z_where(t, ix.sweep): global_store[key.z_where(t, ix.sweep)]()
            for t in range(min(ix.t + 1, T))
        }
        z_what = {key.z_what(z_what_index): z_what_val}
        frame = {"frames": c["frames"]}
        return {**z_wheres, **z_what, **frame}
