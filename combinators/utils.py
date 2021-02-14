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
from combinators.types import check_passable_kwarg, Out
import combinators.tensor.utils as tensor_utils
import combinators.trace.utils as trace_utils
import inspect


@typechecked
def remains_pure(tr: Trace) -> Callable[[Trace], bool]:
    """ TODO """
    cached = tr.detach().deepcopy()  # something like this to detach and copy all tensors

    @typechecked
    def check(tr: Trace) -> None:
        if cached is not None:
            # ensure check that everything is the same
            del cached
            return True
        else:
            raise RuntimeError("You can only perform this check once per operation")

    return check

def save_models(models, filename, weights_dir="./weights"):
    checkpoint = {k: v.state_dict() for k, v in models.items()}

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(model, filename, weights_dir="./weights"):

    checkpoint = torch.load(f'{weights_dir}/{filename}')

    return {k: v.state_dict(checkpoint[k]) for k, v in model.items()}

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

# FIXME: currently not used, but currying the annotations might be nice
def curry(func):
    """ taken from: https://www.python-course.eu/currying_in_python.php """
    curry.__curried_func_name__ = func.__name__
    f_args, f_kwargs = [], {}
    def f(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            return f
        else:
            result = func(*f_args, *f_kwargs)
            f_args, f_kwargs = [], {}
            return result
    return f

