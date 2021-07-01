#!/usr/bin/env python3
from torch import nn, Tensor, optim
from typing import List

from combinators import Propose, Extend, Compose, Resample
from combinators.utils import save_models, load_models, models_as_dict

from experiments.annealing.objectives import stl_trace


def traverse_proposals(fn, out, memo=[]) -> List[Tensor]:
    if out.type == Propose:
        return traverse_proposals(fn, out.q_out, [fn(out)] + memo)
    elif out.type == Extend:
        raise ValueError("impossible! traverse proposal will never arrive here")
    elif out.type == Compose:
        valid_infs = [Compose, Propose, Resample]
        q1_is_inf = out.q1_out.type in valid_infs
        q2_is_inf = out.q2_out.type in valid_infs
        if not (q1_is_inf ^ q2_is_inf):
            return memo
        else:
            return traverse_proposals(fn, out.q1_out if q1_is_inf else out.q2_out, memo)
    elif out.type == Resample:
        return traverse_proposals(fn, out.q_out, memo)
    elif out.type == Condition:
        raise ValueError("impossible! traverse proposal will never arrive here")
    else:
        return memo


def get_stats(out, detach=True):
    fn = lambda out: (
        out.log_weight.detach(),
        out.loss.detach(),
        out.proposal_trace,
        out.target_trace,
        out,
    )
    ret = traverse_proposals(fn, out)
    lws, losses, proposal_trace, target_trace, outs = zip(*ret)
    return dict(
        lw=lws,
        loss=losses,
        proposal_trace=proposal_trace,
        target_trace=target_trace,
        out=outs,
    )


def save_nvi_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    save_models(
        models_as_dict(
            [targets, forwards, reverses], ["targets", "forwards", "reverses"]
        ),
        filename="{}.pt".format(filename),
        weights_dir="./weights",
    )


def load_nvi_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    load_models(
        models_as_dict(
            [targets, forwards, reverses], ["targets", "forwards", "reverses"]
        ),
        filename="./{}.pt".format(filename),
        weights_dir="./weights",
    )
