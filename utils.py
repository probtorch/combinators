#!/usr/bin/env python3

import flatdict

import probtorch
import torch
import torch.nn as nn

def vardict(existing=None):
    vdict = flatdict.FlatDict(delimiter='__')
    if existing:
        for k, v in existing.items():
            vdict[k] = v
    return vdict

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

PARAM_TRANSFORMS = {
    'sigma': nn.functional.softplus,
}

def _parameterize_trace_methods(transforms=PARAM_TRANSFORMS):
    import inspect as _inspect

    for k, v in _inspect.getmembers(probtorch.Trace):
        if _inspect.isfunction(v):
            args = _inspect.signature(v).parameters.keys()
            if 'name' in args and 'value' in args:
                def param_sample(self, params, name=None, value=None, **kwargs):
                    params = {**params[name].copy(), **kwargs}
                    for arg, val in params.items():
                        if arg in transforms:
                            params[arg] = transforms[arg](val)
                    return getattr(self, k)(name=name, value=value, **params)
                setattr(probtorch.Trace, 'param_' + k, param_sample)

_parameterize_trace_methods()
