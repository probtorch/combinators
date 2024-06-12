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
import torch
import sys
import os
import matplotlib.pyplot as plt
from main import nvi_declarative, test, plot_sample_hist, mk_model
from combinators.utils import load_models, models_as_dict
from combinators.objectives import nvo_avo, nvo_rkl
from tqdm import tqdm
from collections import namedtuple


def plot(samples, filename=None):
    K = len(samples)
    fig = plt.figure(figsize=(3 * K, 3 * 1), dpi=300)
    for k in range(K):
        ax4 = fig.add_subplot(1, K, k + 1)
        label, X = samples[k]
        plot_sample_hist(ax4, X, bins=150)
        ax4.set_xlabel(label, fontsize=18)
        if k == 0:
            ax4.set_ylabel("samples", fontsize=18)

    fig.tight_layout(pad=1.0)
    if filename is not None:
        fig.savefig("figures/{}".format(filename), bbox_inches="tight")


def rerun_test(model, objective, filename, yes_plot=False):
    q = nvi_declarative(*model, loss_fn=objective, resample=False)
    losses_test, ess_test, lZ_hat_test, samples_test = test(
        q, (1000, 100), batch_dim=1, sample_dims=0
    )

    if not os.path.exists("./figures/"):
        os.makedirs("./figures/")

    if yes_plot:
        plot(samples_test, filename=filename)
    return losses_test.mean(1), ess_test.mean(1), lZ_hat_test.mean(1)


def get_file_prefix(resample, optimize_path, objective_name, S, K, I):
    return "nvi{}{}_{}_S{}_K{}_I{}".format(
        "r" if resample else "", "s" if optimize_path else "", objective_name, S, K, I
    )


keyspec = namedtuple(
    "keyspec", "resample optimize_path num_targets objective num_iters sample_budget"
)


def from_file_prefix(k):
    runtype, _, objective, ssamples, kk, iiters = k.split("_")
    resampled = runtype[-2] == "r" or runtype[-1] == "r"
    optimize_path = runtype[-1] == "s"
    num_targets = int(kk[1])
    objective = nvo_avo if objective == "avo" else nvo_rkl
    num_iters = int(iiters[1:])
    sample_budget = int(ssamples[1:])
    return keyspec(
        resampled, optimize_path, num_targets, objective, num_iters, sample_budget
    )


def all_file_prefixes():
    S = 288
    I = 20000
    stats = dict()
    for resample in [True, False]:
        for optimize_path in [True, False]:
            for objective_name in ["nvo_avo", "nvo_rkl"]:
                for K in [2, 4, 6, 8]:
                    yield get_file_prefix(
                        resample, optimize_path, objective_name, S, K, I
                    )


def load_metrics(directory):
    stats = dict()
    for prefix in all_file_prefixes():
        key = prefix
        stats[key] = []

        for file in os.listdir(directory):
            if file.startswith(prefix):
                stats[key].append(torch.load(os.path.join(directory, file)))
        if len(stats[key]) == 0:
            del stats[key]
    return stats


def tuple_to_report(losses, esss, lZhs):
    loss = torch.stack(losses)
    ess = torch.stack(esss)
    lZh = torch.stack(lZhs)
    num_seeds = len(losses)
    return dict(
        loss=loss.detach().mean(0),
        ess=ess.mean(0),
        lZh=lZh.mean(0),
        num_seeds=num_seeds,
    )


def main():
    stats = load_metrics("./sync/metrics")
    resampled = []
    all_stats = dict()

    for k, ts in stats.items():
        keyspec = k.split("_")
        runtype, _, objective, ssamples, kk, iiters = [k for k in keyspec]

        if runtype[-2] == "r" or runtype[-1] == "r":
            resampled.append(
                (
                    k,
                    int(kk[1]),
                    runtype[-1] == "s",
                    nvo_avo if objective == "nvo" else nvo_rkl,
                )
            )
        else:
            all_stats[k] = tuple_to_report(*zip(*ts))

    directory = "./sync/weights"
    models = dict()
    for k, num_targets, optimize_path, objective in resampled:
        models[k] = []
        for f in os.listdir(directory):
            if f.startswith(k) and "metric-tuple" not in f and f.endswith("pt"):
                model = mk_model(num_targets, optimize_path=optimize_path)
                load_models(
                    models_as_dict(model, ["targets", "forwards", "reverses"]),
                    filename=f,
                    weights_dir=directory,
                )
                models[k].append((f, model, objective))
    fast = False
    startseed = 5  # nvir high, nvirs on point
    nruns = 1  # FIXME: doesn't work with multiple runs
    for key, smodel in tqdm(models.items(), total=len(models)):
        rerun_stats = []
        for (f, model, objective) in tqdm(smodel, total=len(smodel)):
            for i in range(startseed, startseed + nruns):
                if fast:
                    loss, ess, lZh = [torch.zeros(1) for _ in [None] * 3]
                else:
                    torch.manual_seed(i)
                    loss, ess, lZh = rerun_test(
                        model, objective, filename=f.split(".")[0]
                    )
                rerun_stats.append((loss, ess, lZh))
        all_stats[key] = tuple_to_report(*zip(*rerun_stats))

    for k, v in all_stats.items():
        print(k, v["num_seeds"])

    for key, metrics in all_stats.items():
        runtype, _, o, ssamples, kk, iiters = key.split("_")
        print("\n", runtype, o, int(kk[1]))

        for k, v in metrics.items():
            if k != "loss":
                print("    ", k, v)

    table = [("method", "#", "num_targets", "ess", "lZh")]
    for key, metrics in all_stats.items():
        runtype, _, objective, ssamples, kk, iiters = key.split("_")
        show_float = lambda s: "{:.4f}".format(s)
        table.append(
            (
                "avo" if objective == "avo" else runtype,
                str(metrics["num_seeds"]),
                kk[1],
                show_float(metrics["ess"][-1].item()),
                show_float(metrics["lZh"][-1].item()),
            )
        )

    max_col = []

    for col in zip(*table):
        max_col.append(max(map(len, col)))

    last = None
    for i, row in enumerate(table):
        row_str = ""
        for j, colmax in enumerate(max_col):

            if j != 0:  # (i == 0 and j != 0) or j == 1:
                row_str += "| "
            item = ("{:>" + str(colmax) + "}").format(row[j])
            if j == 0:
                if last is None:
                    last = item
                if item != last:
                    print((sum(max_col) + len(max_col) * 3) * "-")
                last = item

            row_str += item + " "
        print(row_str)

    print()


if __name__ == "__main__":
    main()
