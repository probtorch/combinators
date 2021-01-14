#!/usr/bin/env python3
import torch
from torch import Tensor
from combinators.stochastic import Trace, Provenance, RandomVariable, ImproperRandomVariable
from combinators.types import TraceLike
from typing import Callable, Any, Tuple, Optional, Set
from copy import deepcopy
from typeguard import typechecked
from itertools import chain
from typing import *
import combinators.tensor.utils as tensor_utils
from enum import Enum, auto

@typechecked
def is_valid_subtype(_subtr: Trace, _super: Trace)->bool:
    """ sub trace <: super trace """
    super_keys = frozenset(_super.keys())
    subtr_keys = frozenset(_subtr.keys())

    @typechecked
    def check(key:str)->bool:
        try:
            return _subtr[key].value.shape == _super[key].value.shape
        except:  # FIXME better capture
            return False

    # FIXME: add better error message about dimension checks
    return len(super_keys - subtr_keys) == 0 and all([check(key) for key in super_keys])

@typechecked
def assert_valid_subtrace(tr1:Trace, tr2:Trace) -> None:
    assert is_valid_subtype(tr1, tr2), "{} is not a subtype of {}".format(tr1, tr2)


@typechecked
def valeq(t1:Trace, t2:Trace, nodes:Optional[Dict[str, Any]]=None, check_exist:bool=True)->bool:
    """
    Two traces are value-equivalent for a set of names iff the values of the corresponding names in both traces exist
    and are mutually equal.

    check_exist::boolean    Controls if a name must exist in both traces, if set to False one values of variables that
                            exist in both traces are checks for 'value equivalence'
    """
    if nodes is None:
        nodes = set(t1._nodes.keys()).intersection(t2._nodes.keys())
    # pretty sure this is all you need:
    all_in = all(name in t1 and name in t2 for name in nodes)
    all_eq = all(trace_eq(t1, t2, name) for name in nodes)
    # this is the original code:
    for name in nodes:
        # TODO: check if check_exist handes everything correctly
        if t1[name] is not None and t2[name] is not None:
            if not torch.equal(t1[name].value, t2[name].value):
                if all_eq:
                    import ipdb; ipdb.set_trace();
                    raise Exception("Check logic!!!")
                return False
                # raise Exception("Values for same RV differs in traces")
        elif check_exist:
            if not all_in:
                import ipdb; ipdb.set_trace();
                raise Exception("Check logic!!!")
            raise Exception("RV does not exist in traces")

    if not (all_in and all_eq):
        import ipdb; ipdb.set_trace();
        raise Exception("Check logic!!!")

    return True


@typechecked
def show(tr:TraceLike, fix_width=False):
    get_value = lambda v: v if isinstance(v, Tensor) else v.value
    ten_show = lambda v: tensor_utils.show(get_value(v), fix_width=fix_width)
    return "{" + "; ".join([f"'{k}'-➢{ten_show(v)}" for k, v in tr.items()]) + "}"

@typechecked
def trace_eq(t0:Trace, t1:Trace, name:str):
    return name in t0 and name in t1 and torch.equal(t0[name].value, t1[name].value)

@typechecked
def log_importance_weight(proposal_trace:Trace, target_trace:Trace, batch_dim:int=1, sample_dims:int=0, check_exist:bool=True)->Tensor:
    """
    Computes an importance weight by substractiong log joint distibution of the proposal trace from the log joint
    distributin of the target trace. The log joint is computed based on all nodes inside the target trace.

    NOTE: Important weights can only be computes for value-equivalent traces.
    NOTE: Implicit proposals are not supported, i.e. all names in the target trace must exist in the proposal trace.
    TODO: Fix batching
    """
    assert valeq(proposal_trace, target_trace, nodes=target_trace._nodes, check_exist=check_exist)
    return target_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes) - \
        proposal_trace.log_joint(batch_dim=batch_dim, sample_dims=sample_dims, nodes=target_trace._nodes)


def copysubtrace(tr: Trace, subset: Optional[Set[str]]):
    # FIXME: need to verify that this does the expected thing
    out = Trace()
    for key, node in tr.items():
        if subset is None or key in subset:
            out[key] = node
    return out


class RequiresGrad(Enum):
    YES=auto()
    NO=auto()
    DEFAULT=auto()

def get_requires_grad(ten, is_detach_set, global_requires_grad:RequiresGrad=RequiresGrad.DEFAULT):
    if is_detach_set:
        return False
    else:
        return ten.requires_grad if global_requires_grad==RequiresGrad.DEFAULT else global_requires_grad == RequiresGrad.YES

@typechecked
def copytraces(*traces: Trace, requires_grad:RequiresGrad=RequiresGrad.DEFAULT, detach:Set[str]=set(), overwrite=False)->Trace:
    """
    shallow-copies nodes from many traces into a new trace.
    unless overwrite is set, there is a first-write presidence.
    """
    newtr = Trace()
    for tr in traces:
        for k, rv in tr.items():
            if k in newtr:
                if overwrite:
                    raise NotImplementedError("")
                else:
                    pass

            newrv = copyrv(rv, requires_grad=get_requires_grad(rv.value, k in detach, requires_grad))
            newtr.append(newrv, name=k)
    return newtr

@typechecked
def copytrace(tr: Trace, **kwargs)->Trace:
    # sugar
    return copytraces(tr, **kwargs)

@typechecked
def copyrv(rv:Union[RandomVariable, ImproperRandomVariable], requires_grad: bool=True, provenance:Optional[Provenance]=None, deepcopy_value=False):
    RVClass = type(rv)
    value = tensor_utils.copy(rv.value, requires_grad=requires_grad, deepcopy=deepcopy_value)
    provenance = provenance if provenance is not None else rv.provenance

    if RVClass is RandomVariable:
        # TODO: what about copying the dist?
        return RVClass(dist=rv.dist, value=value, provenance=provenance, mask=rv.mask, use_pmf=rv._use_pmf)
    elif RVClass is ImproperRandomVariable:
        return RVClass(log_density_fn=rv._log_density_fn, value=value, provenance=provenance, mask=rv.mask)
    else:
        raise NotImplementedError()
