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
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def plot_sample_hist(
    ax, samples, sort=True, bins=50, range=None, weight_cm=False, **kwargs
):
    ax.grid(False)
    samples = torch.flatten(samples[:, :10], end_dim=-2)
    x, y = samples.detach().cpu().numpy().T
    mz, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True, range=range)
    X, Y = np.meshgrid(x_e, y_e)
    if weight_cm:
        raise NotImplemented()
    else:
        ax.grid(False)
        ax.imshow(mz, **kwargs)


def plot(losses, ess, lZ_hat, samples, filename=None):
    K = losses.shape[1]
    fig = plt.figure(figsize=(K * 4, 3 * 4), dpi=300)
    for k in range(K):
        ax1 = fig.add_subplot(4, K, k + 1)
        ax1.plot(losses[:, k].detach().cpu().squeeze())
        if k == 0:
            ax1.set_ylabel("loss", fontsize=18)
        ax2 = fig.add_subplot(4, K, k + 1 + K)
        ax2.plot(ess[:, k].detach().cpu().squeeze())
        if k == 0:
            ax2.set_ylabel("ess", fontsize=18)
        ax3 = fig.add_subplot(4, K, k + 1 + 2 * K)
        ax3.plot(lZ_hat[:, k].detach().cpu().squeeze())
        if k == 0:
            ax3.set_ylabel("log_Z_hat", fontsize=18)

    for k in range(K):
        ax4 = fig.add_subplot(4, K, k + 1 + 3 * K)
        label, X = samples[k]
        plot_sample_hist(ax4, X, bins=150)
        ax4.set_xlabel(label, fontsize=18)
        if k == 0:
            ax4.set_ylabel("samples", fontsize=18)

    fig.tight_layout(pad=1.0)
    if filename is not None:
        fig.savefig("figures/{}".format(filename), bbox_inches="tight")
