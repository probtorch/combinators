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
import os
import subprocess
import random
import torch
import numpy as np
from torch import Tensor, optim
import torch.distributions as D
from probtorch.stochastic import Trace, RandomVariable
from typing import Any
from typeguard import typechecked
from combinators.out import Out
import combinators.tensor.utils as tensor_utils
import combinators.trace.utils as trace_utils


def save_models(models, filename, weights_dir="./weights") -> None:
    checkpoint = {k: v.state_dict() for k, v in models.items()}

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    torch.save(checkpoint, f"{weights_dir}/{filename}")


def load_models(model, filename, weights_dir="./weights", **kwargs) -> None:
    path = os.path.normpath(f"{weights_dir}/{filename}")

    checkpoint = torch.load(path, **kwargs)

    {k: v.load_state_dict(checkpoint[k]) for k, v in model.items()}


def models_as_dict(model_iter, names):
    """(for annealing) given a list of list of targets and kernels -- flatten for save_models and load_models, above"""
    assert isinstance(model_iter, (tuple, list)) or all(
        map(lambda ms: isinstance(ms, (tuple, list)), model_iter.values())
    ), "takes a list or dict of lists"
    assert len(names) == len(model_iter), "names must exactly align with model lists"

    model_dict = dict()
    for i, (name, models) in enumerate(
        zip(names, model_iter)
        if isinstance(model_iter, (tuple, list))
        else model_iter.items()
    ):
        for j, m in enumerate(models):
            model_dict[f"{str(i)}_{name}_{str(j)}"] = m
    return model_dict


def adam(models, **kwargs):
    """Adam for dicts or iterables of models"""
    iterable = models.values() if isinstance(models, dict) else models
    return optim.Adam([dict(params=x.parameters()) for x in iterable], **kwargs)


def git_root() -> str:
    """print the root of the project"""
    return (
        subprocess.check_output("git rev-parse --show-toplevel", shell=True)
        .decode("utf-8")
        .rstrip()
    )


def ppr_show(a: Any, m: str = "dv", debug: bool = False, **kkwargs: Any):
    """show instance of a prettified object"""
    if debug:
        print(type(a))
    if isinstance(a, Tensor):
        return tensor_utils.show(a)
    elif isinstance(a, D.Distribution):
        return trace_utils.showDist(a)
    elif isinstance(a, list):
        return "[" + ", ".join(map(ppr_show, a)) + "]"
    elif isinstance(a, (Trace, RandomVariable)):
        args = []
        kwargs = dict()
        if m is not None:
            if "v" in m or m == "a":
                args.append("value")
            if "p" in m or m == "a":
                args.append("log_prob")
            if "d" in m or m == "a":
                kwargs["dists"] = True
        showinstance = (
            trace_utils.showall if isinstance(a, Trace) else trace_utils.showRV
        )
        if debug:
            print("showinstance", showinstance)
            print("args", args)
            print("kwargs", kwargs)
        return showinstance(a, args=args, **kwargs, **kkwargs)
    elif isinstance(a, Out):
        print(f"got type {type(a)}, guessing you want the trace:")
        return ppr_show(a.trace)
    elif isinstance(a, dict):
        return repr({k: ppr_show(v) for k, v in a.items()})
    else:
        return repr(a)


def ppr(a: Any, m="dv", debug=False, desc="", **kkwargs):
    """a pretty printer that relies ppr_show"""
    print(desc, ppr_show(a, m=m, debug=debug, **kkwargs))


@typechecked
def runtime() -> str:
    try:
        # magic global function in ipython shells
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
        else:
            raise Exception()
    except:
        return "terminal"


@typechecked
def seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.benchmark = (
        True  # just incase something goes wrong with set_deterministic
    )
    if torch.__version__[:3] == "1.8":
        pass
        # torch.use_deterministic_algorithms(True)


def exit0():
    try:
        import pytest

        xit = pytest.exit
    except:
        import sys

        xit = sys.exit
    xit(0)


def is_smoketest() -> bool:
    env_var = os.getenv("SMOKE")
    return env_var is not None and env_var == "true"


def propagate(
    N: D.MultivariateNormal,
    F: Tensor,
    t: Tensor,
    B: Tensor,
    marginalize: bool = False,
    reverse_order: bool = False,
) -> D.MultivariateNormal:
    # N is normal starting from
    F = F.cpu()  # F is NN weights on linear network of forward kernel
    t = t.cpu()  # t is bias
    B = B.cpu()  # b is cov of kernel
    with torch.no_grad():
        a = N.loc.cpu()
        A = N.covariance_matrix.cpu()
        b = t + F @ a
        m = torch.cat((a, b))
        FA = F @ A
        BFFA = B + F @ (FA).T
        if marginalize:
            return D.MultivariateNormal(loc=b, covariance_matrix=BFFA)
        if not reverse_order:
            A = N.covariance_matrix.cpu()
            C1 = torch.cat((A, (FA).T), dim=1)
            C2 = torch.cat((FA, BFFA), dim=1)
            C = torch.cat((C1, C2), dim=0)
        if reverse_order:
            C1 = torch.cat((BFFA, FA), dim=1)
            C2 = torch.cat(((FA).T, A), dim=1)
            C = torch.cat((C1, C2), dim=0)
            m = torch.cat((b, a))
        return D.MultivariateNormal(loc=m, covariance_matrix=C)
