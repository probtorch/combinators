#!/usr/bin/env python3

import collections
from distutils.version import LooseVersion
import flatdict

import probtorch
import torch
import torch.nn as nn
import torch.onnx

EMPTY_TRACE = collections.defaultdict(lambda: None)

def optional_to(tensor, other):
    if isinstance(tensor, probtorch.stochastic.RandomVariable):
        return tensor.value.to(other)
    elif tensor is not None:
        return tensor.to(other)
    return tensor

def particle_index(tensor, indices):
    indexed_tensors = [t[indices[particle]] for particle, t in
                       enumerate(torch.unbind(tensor, 0))]
    return torch.stack(indexed_tensors, dim=0)

def map_tensors(f, *args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            yield f(arg)
        else:
            yield arg

def vardict(existing=None):
    vdict = flatdict.FlatDict(delimiter='__')
    if existing:
        for k, v in existing.items():
            vdict[k] = v
    return vdict

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

def walk_trie(trie, keys=[]):
    while len(keys) > 0:
        trie = trie[keys[0]]
        keys = keys[1:]
    return trie

PARAM_TRANSFORMS = {
    'scale': nn.functional.softplus,
}

def _parameterize_trace_methods(transforms=PARAM_TRANSFORMS):
    import inspect as _inspect

    for k, v in _inspect.getmembers(probtorch.Trace):
        if _inspect.isfunction(v):
            args = _inspect.signature(v).parameters.keys()
            if 'name' in args and 'value' in args:
                def param_sample(self, params, name=None, value=None, k=k,
                                 **kwargs):
                    params = {**params[name].copy(), **kwargs}
                    for arg, val in params.items():
                        if arg in transforms:
                            params[arg] = transforms[arg](val)
                    return getattr(self, k)(name=name, value=value, **params)
                setattr(probtorch.Trace, 'param_' + k, param_sample)

_parameterize_trace_methods()


def graph(model, args, verbose=False):
    with torch.onnx.set_training(model, False):
        try:
            trace, _ = torch.jit.get_trace_graph(model, args)
        except RuntimeError:
            print('Error occurs, No graph saved')
            _ = model(args)  # don't catch, just print the error message
            print("Checking if it's onnx problem...")
            try:
                import tempfile
                torch.onnx.export(model, args, tempfile.TemporaryFile(),
                                  verbose=True)
            except RuntimeError:
                print("Your model fails onnx too, please report to onnx team")
            return None
    if LooseVersion(torch.__version__) >= LooseVersion('0.4'):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    result = trace.graph()
    if verbose:
        print(result)
    return result
