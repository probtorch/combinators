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
from tqdm import trange
from combinators import adam, effective_sample_size, log_Z_hat

from experiments.annealing.models import mk_model
from experiments.annealing.objectives import nvo_rkl, nvo_rkl_mod, nvo_avo
from experiments.annealing.plotting import plot
from experiments.annealing.utils import get_stats, save_nvi_model, load_nvi_model
from experiments.annealing.vsmc import vsmc


def nvi_train(
    q,
    targets,
    forwards,
    reverses,
    sample_shape=(11,),
    batch_dim=1,
    sample_dims=0,
    iterations=100,
):

    # get shapes for metrics aggregation
    stats_nvi_run = get_stats(
        q(
            None,
            sample_shape=sample_shape,
            batch_dim=batch_dim,
            sample_dims=sample_dims,
            _debug=True,
        )
    )
    mk_metric = lambda kw: torch.zeros(
        iterations, len(stats_nvi_run[kw]), *stats_nvi_run[kw][0].shape
    )
    full_losses, full_lws = mk_metric("loss"), mk_metric("lw")

    # Check inizialization
    optimizer = adam([*targets, *forwards, *reverses])

    tqdm_iterations = trange(iterations)
    for i in tqdm_iterations:
        out = q(
            None,
            sample_shape=sample_shape,
            batch_dim=batch_dim,
            sample_dims=sample_dims,
            _debug=True,
        )
        loss = out.loss.mean()
        lw = out.q_out.log_weight if out.type == "Resample" else out.log_weight
        ess = effective_sample_size(lw, sample_dims=sample_dims)
        lZ_hat = log_Z_hat(lw, sample_dims=sample_dims)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =================================================== #
        #                    tqdm updates                     #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        tqdm_iterations.set_postfix(
            loss="{:09.4f}".format(loss.item()),
            ess="{:09.4f}".format(ess.item()),
            log_Z_hat="{:09.4f}".format(lZ_hat.item()),
        )
        # =================================================== #
        stats_nvi_run = get_stats(out)
        lw, loss, _, _ = (
            stats_nvi_run["lw"],
            stats_nvi_run["loss"],
            stats_nvi_run["proposal_trace"],
            stats_nvi_run["target_trace"],
        )
        torch.stack(loss, dim=0, out=full_losses[i])
        torch.stack(lw, dim=0, out=full_lws[i])

    ess = effective_sample_size(full_lws, sample_dims=2)
    lZ_hat = log_Z_hat(full_lws, sample_dims=2)
    return q, full_losses, ess, lZ_hat


def nvi_test(vsmc_program, sample_shape, batch_dim=1, sample_dims=0):
    out = vsmc_program(
        None,
        sample_shape=sample_shape,
        sample_dims=sample_dims,
        batch_dim=batch_dim,
        _debug=True,
    )
    stats_nvi_run = get_stats(out)
    lw = torch.stack(stats_nvi_run["lw"])
    loss = torch.stack(stats_nvi_run["loss"])
    proposal_traces = stats_nvi_run["proposal_trace"]

    ess = effective_sample_size(lw, sample_dims=sample_dims + 1)
    lZ_hat = log_Z_hat(lw, sample_dims=sample_dims + 1)
    samples = [
        ("g{}".format(t + 1), proposal_traces[t]["g{}".format(t + 1)].value)
        for t in range(len(proposal_traces))
    ]  # skip the initial gaussian proposal
    return loss, ess, lZ_hat, samples, out


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser("Combinators annealing stats")
    # data config
    parser.add_argument("--objective", default="nvo_rkl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resample", default=False, type=bool)
    parser.add_argument("--iterations", default=20000, type=int)
    parser.add_argument("--num_targets", default=4, type=int)
    parser.add_argument("--sample_budget", default=288, type=int)
    parser.add_argument("--optimize_path", default=False, type=bool)
    parser.add_argument("--train_and_test", default=True, type=bool)

    # CI config
    parser.add_argument("--smoketest", default=False, type=bool)

    args = parser.parse_args()

    S = args.sample_budget
    K = args.num_targets
    seed = args.seed
    resample = args.resample
    iterations = args.iterations
    optimize_path = args.optimize_path
    tt = args.train_and_test

    if not os.path.exists("./weights/"):
        os.makedirs("./weights/")

    if not os.path.exists("./metrics/"):
        os.makedirs("./metrics/")

    if args.objective == "nvo_avo":
        objective = nvo_avo
    elif args.objective == "nvo_rkl":
        objective = nvo_rkl
    elif args.objective == "nvo_rkl_mod":
        objective = nvo_rkl_mod
    else:
        raise TypeError(
            "objective is one of: {}".format(", ".join(["nvo_avo", "nvo_rkl"]))
        )

    save_plots = resample and optimize_path and K == 8
    filename = "nvi{}{}_{}_S{}_K{}_I{}_seed{}".format(
        "r" if resample else "",
        "s" if optimize_path else "",
        objective.__name__,
        S,
        K,
        iterations,
        seed,
    )

    torch.manual_seed(seed)
    model = mk_model(K, optimize_path=optimize_path)
    q = vsmc(*model, objective, resample=resample)
    S = 288
    iterations = args.iterations

    losses, ess, lZ_hat = (
        torch.zeros(iterations, K - 1, 1),
        torch.zeros(iterations, K - 1, 1),
        torch.zeros(iterations, K - 1, 1),
    )

    if tt:
        q, losses, ess, lZ_hat = nvi_train(
            q,
            *model,
            sample_shape=(S // K, 1),
            iterations=iterations,
            batch_dim=1,
            sample_dims=0,
        )
        save_nvi_model(*model, filename=filename)
    else:
        load_nvi_model(*model, filename=filename)

    q = vsmc(*model, objective, resample=False)

    if not args.smoketest:
        losses_test, ess_test, lZ_hat_test, samples_test, _ = nvi_test(
            q, (1000, 100), batch_dim=1, sample_dims=0
        )

        torch.save(
            (losses_test.mean(1), ess_test.mean(1), lZ_hat_test.mean(1)),
            "./metrics/{}-metric-tuple_S{}_B{}-loss-ess-logZhat.pt".format(
                filename, 1000, 100
            ),
        )

        print("losses:", losses_test.mean(1))
        print("ess:", ess_test.mean(1))
        print("log_Z_hat", lZ_hat_test.mean(1))

        if save_plots:
            if not os.path.exists("./figures/"):
                os.makedirs("./figures/")

            plot(losses, ess, lZ_hat, samples_test, filename=filename)

    print("done!")
