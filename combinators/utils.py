#!/usr/bin/env python3
import torch
from probtorch import Trace
from typing import Callable, Any, Tuple, Optional, Set
from copy import deepcopy

def is_valid_subtype(_subtr: Trace, _super: Trace):
    """ sub trace <: super trace """
    super_keys = frozenset(_super.keys())
    subtr_keys = frozenset(_subtr.keys())

    def check(key):
        try:
            return _subtr[key].shape == _super[key].shape
        except:  # FIXME better capture
            return False

    return len(super_keys - subtr_keys) == 0 and all([check(key) for key in super_keys])


def assert_valid_subtrace(prog1, prog2):
    assert is_valid_subtype(prog1.trace, prog2.trace), "{} is not a subtype of {}".format(prog1.trace, prog2.trace)


def remains_pure(tr: Trace):
    """ TODO """
    cached = tr.detach().deepcopy()  # something like this to detach and copy all tensors

    def check(tr: Trace):
        if cached is not None:
            # ensure check that everything is the same
            del cached
            return True
        else:
            raise RuntimeError("You can only perform this check once per operation")

    return check


def valeq(t1, t2, nodes=None, check_exist=True):
    """
    Two traces are value-equivalent for a set of names iff the values of the corresponding names in both traces exist
    and are mutually equal.

    check_exist::boolean    Controls if a name must exist in both traces, if set to False one values of variables that
                            exist in both traces are checks for 'value equivalence'
    """
    if nodes is None:
        nodes = set(t1._nodes.keys()).intersection(t2._nodes.keys())
    for name in nodes:
        # TODO: check if check_exist handes everything correctly
        if t1[name] is not None and t2[name] is not None:
            if not torch.equal(t1[name].value, t2[name].value):
                return False
                # raise Exception("Values for same RV differs in traces")
        elif check_exist:
            raise Exception("RV does not exist in traces")
    return True


def copytrace(tr: Trace, subset: Optional[Set[str]]):
    out = Trace()
    for key, node in tr:
        if subset is None or key in subset:
            out[key] = node.detach().copy()
    return out


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
