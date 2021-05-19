#!/usr/bin/env python3
import torch
from torch import Tensor
from enum import Enum, auto, unique
from torch import distributions as D
from typing import Callable, Any, Tuple, Optional, Set, Union, Dict, List
from typeguard import typechecked

from probtorch.stochastic import Trace, Provenance, _RandomVariable, RandomVariable, ImproperRandomVariable
import inspect

import combinators.tensor.utils as tensor_utils
from combinators.program import check_passable_kwarg

TraceLike = Union[Trace, Dict[str, Union[Tensor, _RandomVariable]]]


def distprops(dist):
    """ return a list of a distribution's properties """
    return [
        p for p in (set(inspect.getfullargspec(dist.__init__).args) - {'self'})
            if hasattr(dist, p)
    ]

def distprops_to_kwargs(dist):
    return {p: getattr(dist, p) for p in distprops(dist)}

@typechecked
def maybe_sample(trace:Optional[Trace], sample_shape:Union[Tuple[int], None, tuple], reparameterized:Optional[bool]=None) -> Callable[[D.Distribution, str], Tuple[Tensor, Provenance, bool]]:

    def curried(dist:D.Distribution, name:str) -> Tuple[Tensor, Provenance, bool]:
        _reparameterized = reparameterized if reparameterized is not None else dist.has_rsample
        if trace is not None and name in trace:
            return trace[name].value, Provenance.REUSED, _reparameterized
        elif sample_shape is None:
            return (dist.rsample() if _reparameterized else dist.sample(), Provenance.SAMPLED, _reparameterized)
        else:
            return (dist.rsample(sample_shape) if _reparameterized else dist.sample(sample_shape), Provenance.SAMPLED, _reparameterized)
    return curried


@typechecked
def valeq(t1:Trace, t2:Trace, nodes:Union[Dict[str, Any], List[Tuple[str, str]], None]=None, check_exist:bool=True, strict:bool=False)->bool:
    """
    Two traces are value-equivalent for a set of names iff the values of the corresponding names in both traces exist
    and are mutually equal.

    check_exist::boolean    Controls if a name must exist in both traces, if set to False one values of variables that
                            exist in both traces are checks for 'value equivalence'
    """
    t1nodes:Set[str] = set(t1._nodes.keys())
    t2nodes:Set[str] = set(t2._nodes.keys())
    if nodes is None:
        _nodes = t1nodes.union(t2nodes) if strict else t1nodes.intersection(t2nodes)
    elif isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], tuple) and len(nodes[0]) == 2:
        raise NotImplementedError('pairing tuples not supported yet')
    else:
        _nodes = nodes

    if not all(name in t1 and name in t2 for name in _nodes):
        str_t1nodes = "{" + ", ".join(t1nodes) + "}"
        str_t2nodes = "{" + ", ".join(t2nodes) + "}"
        cmpr_string = "are not in {" + ", ".join(_nodes) + "}" if nodes is not None \
            else "are not equal"
        raise Exception("trace's keys {}:\n  trace 1: {}\n  trace 2: {}".format(cmpr_string, str_t1nodes, str_t2nodes))

    invalid = list(filter(lambda name: not trace_eq(t1, t2, name), _nodes))
    if len(invalid) > 0:
        str_nodes = "{" + ", ".join(invalid) + "}"
        raise Exception("RV nodes {} are not equal in:\n  trace 1: {}\n  trace 2: {}".format(str_nodes, set(t1.keys()), set(t2.keys())))

    return True


@typechecked
def show(tr:TraceLike, fix_width=False):
    """ show a trace """
    get_value = lambda v: v if isinstance(v, Tensor) else v.value
    ten_show = lambda v: tensor_utils.show(get_value(v), fix_width=fix_width)
    return "{" + "; ".join([f"'{k}'-➢{ten_show(v)}" for k, v in tr.items()]) + "}"


def showDist(dist):
    ''' prettier show instance for distributions more information when debugging '''
    props = distprops(dist)
    sattrs = [f'{p}:{tensor_utils.show(getattr(dist, p))}' for p in props]
    return type(dist).__name__ + "(" +", ".join(sattrs)+ ")"


def showRV(v, args=[], dists=False):
    ''' prettier show instance for RandomVariables with more information when debugging '''
    if len(args) == 0 and not dists:
        print("[WARNING] asked to show a RV, but no arguments passed")
    name = ""
    arglist = [f'{a}={tensor_utils.show(getattr(v, a))}' for a in args]
    if dists:
        name = type(v.dist).__name__ + "("
        props = distprops(v.dist)
        sattrs = [f'{p}={tensor_utils.show(getattr(v.dist, p))}' for p in props]
        arglist.append('dist=' + name + "(" + ", ".join(sattrs) + ")")
    return name + ", ".join(arglist)


@typechecked
def showall(tr:Trace, delim="; ", pretty=True, mlen=0, sort=True, args=[], dists=False):
    """ show instance for a trace, using showRV """
    items = list(tr.items())
    if len(items) == 0:
        return "Trace{}"
    if pretty:
        mlen = max(map(len, tr.keys()))
        delim = "\n,"
    if sort:
        items.sort()
    pref = delim if "\n" == delim[0] or "\n" == delim[-1] else ""

    return "{" + pref + delim.join([("{:>"+str(mlen)+"}-➢{}").format(k, showRV(v, args=args, dists=dists)) for k, v in items]) + "}"


@typechecked
def showvals(tr:Trace, delim="; ", pretty=True, mlen=0, sort=True):
    return showall(tr, args=['value'])


@typechecked
def showprobs(tr:Trace, delim="; ", pretty=True, mlen=0, sort=True):
    return showall(tr, args=['log_prob'])


@typechecked
def showdists(tr:Trace, delim="; ", pretty=True, mlen=0, sort=True):
    return showall(tr, args=[], dists=True)


@typechecked
def trace_eq(t0:Trace, t1:Trace, name:str):
    return name in t0 and name in t1 and torch.equal(t0[name].value, t1[name].value)



@unique
class WriteMode(Enum):
    LastWriteWins = auto()
    FirstWriteWins = auto()
    NoOverlaps = auto()


def copytraces(*traces, exclude_nodes=None, mode=WriteMode.NoOverlaps):
    """
    merge traces together. domains should be disjoint otherwise last-write-wins.
    """
    newtrace = Trace()
    if exclude_nodes is None:
        exclude_nodes = {}

    for tr in traces:
        for k, rv in tr.items():
            if k in exclude_nodes:
                continue
            elif k in newtrace:
                if mode == WriteMode.LastWriteWins:
                    newtrace._nodes[k] = tr[k]
                elif mode == WriteMode.FirstWriteWins:
                    continue
                elif mode == WriteMode.NoOverlaps:
                    raise RuntimeError("traces should not overlap")
                else:
                    raise TypeError("impossible specification")

            newtrace._inject(rv, name=k, silent=True)
    return newtrace


def rerun_with_detached_values(trace: Trace):
    """
    Rerun a trace with detached values, recomputing the computation graph so that
    value do not cause a gradient leak.
    """
    newtrace = Trace()

    def rerun_rv(rv):
        value = rv.value.detach()
        if isinstance(rv, RandomVariable):
            return RandomVariable(
                value=value,
                dist=rv.dist,
                provenance=rv.provenance,
                reparameterized=rv.reparameterized,
            )
        elif isinstance(rv, ImproperRandomVariable):
            return ImproperRandomVariable(
                value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance
            )
        else:
            raise NotImplementedError(
                "Only supports RandomVariable and ImproperRandomVariable"
            )

    for k, v in trace.items():
        newtrace._inject(rerun_rv(v), name=k)

    return newtrace
