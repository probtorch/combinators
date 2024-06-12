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
from matplotlib import gridspec, patches
import matplotlib.pyplot as plt


def viz_samples(
    frames,
    rs,
    ws,
    num_sweeps,
    num_objects,
    object_size,
    fs=1,
    title_fontsize=12,
    lw=2,
    colors=[
        "#AA3377",
        "#EE7733",
        "#009988",
        "#0077BB",
        "#BBBBBB",
        "#EE3377",
        "#DDCC77",
    ],
    save=False,
):
    B, T, FP, _ = frames.shape
    recons_last = rs[0]
    z_wheres_last = ws[0].clone()
    z_wheres_last[:, :, :, 1] = z_wheres_last[:, :, :, 1] * (-1)
    c_pixels_last = z_wheres_last
    c_pixels_last = (c_pixels_last + 1.0) * (FP - object_size) / 2.0  # B * T * K * D
    for b in range(B):
        num_cols = T
        num_rows = 2
        c_pixel_last, recon_last = c_pixels_last[b].numpy(), recons_last[b].numpy()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05
        )
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for c in range(num_cols):
            ax_infer = fig.add_subplot(gs[0, c])
            ax_infer.imshow(frames[b, c].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax_infer.set_xticks([])
            ax_infer.set_yticks([])
            for k in range(num_objects):
                rect_k = patches.Rectangle(
                    (c_pixel_last[c, k, :]),
                    object_size,
                    object_size,
                    linewidth=lw,
                    edgecolor=colors[k],
                    facecolor="none",
                )
                ax_infer.add_patch(rect_k)
            ax_recon = fig.add_subplot(gs[1, c])
            ax_recon.imshow(recon_last[c], cmap="gray", vmin=0.0, vmax=1.0)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
    if save:
        plt.savefig("combinators_apg_samples.svg", dpi=300)
