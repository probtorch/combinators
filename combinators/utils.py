#!/usr/bin/env python3
import os
import subprocess
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.distributions as D
from combinators.stochastic import Trace, RandomVariable
from typing import Callable, Any, Tuple, Optional, Set
from copy import deepcopy
from typeguard import typechecked
from combinators.out import Out
from combinators.program import check_passable_kwarg, Out
import combinators.tensor.utils as tensor_utils
import combinators.trace.utils as trace_utils
import inspect


def save_models(models, filename, weights_dir="./weights")->None:
    checkpoint = {k: v.state_dict() for k, v in models.items()}

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(model, filename, weights_dir="./weights", **kwargs)->None:
    path = os.path.normpath(f'{weights_dir}/{filename}')

    checkpoint = torch.load(path, **kwargs)

    {k: v.load_state_dict(checkpoint[k]) for k, v in model.items()}

def models_as_iter(model_dict, with_names=False):
    from collections import namedtuple
    model_spec = namedtuple('model_spec', ['group_ix', 'model_ix', 'group'])

    def to_model_spec(name):
        parts = name.split("_")
        group_order = int(parts[0])
        model_order = int(parts[-1])
        group = "_".join(parts[1:-1])
        return model_spec(group_order, model_order, group)

    def from_model_spec(spec):
        return f'{spec.group_ix}_{spec.group}_{spec.model_ix}'

    spec = {to_model_spec(name) for name in model_dict.keys()}
    group_names = {(group, gix) for (gix, mix, group) in spec}
    groups = [None] * len(group_names)
    names = [None] * len(group_names)
    model_len = None

    for g, gix in group_names:
        gspec = list(filter(lambda t: t.group==g, spec))
        _model_len = max(map(lambda t: t.model_ix, gspec))
        if model_len is None:
            model_len = _model_len
        assert model_len == _model_len

        models = [None]*model_len
        for mspec in gspec:
            model_key = from_model_spec(mspec)
            models[mspec.model_ix] = model_dict[model_key]

        groups[gix] = models
        names[gix] = g

    return (groups, names) if with_names else groups

def models_as_dict(model_iter, names):
    assert isinstance(model_iter, (tuple, list)) or all(map(lambda ms: isinstance(ms, (tuple,list)), model_iter.values())), "takes a list or dict of lists"
    assert len(names) == len(model_iter), 'names must exactly align with model lists'

    model_dict = dict()
    for i, (name, models) in enumerate(zip(names, model_iter) if isinstance(model_iter, (tuple, list)) else model_iter.items()):
        for j, m in enumerate(models):
            model_dict[f'{str(i)}_{name}_{str(j)}'] = m
    return model_dict

def adam(models, **kwargs):
    iterable = models.values() if isinstance(models, dict) else models
    return optim.Adam([dict(params=x.parameters()) for x in iterable], **kwargs)

def git_root():
    return subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode("utf-8").rstrip()

def ppr_show(a:Any, m='dv', debug=False, **kkwargs):
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
            if 'v' in m or m == 'a':
                args.append('value')
            if 'p' in m or m == 'a':
                args.append('log_prob')
            if 'd' in m or m == 'a':
                kwargs['dists'] = True
        showinstance = trace_utils.showall if isinstance(a, Trace) else trace_utils.showRV
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

def ppr(a:Any, m='dv', debug=False, desc='', **kkwargs):
    print(desc, ppr_show(a, m=m, debug=debug, **kkwargs))

def pprm(a:Tensor, name='', **kkwargs):
    ppr(a, desc="{} ({: .4f})".format(name, a.detach().cpu().mean().item()), **kkwargs)

