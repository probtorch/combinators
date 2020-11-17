#!/usr/bin/env python3

from pytorch import Tensor
from pytorch.distributions import Distribution
from probtorch import Trace
from typing import Union, List
from abc import ABC
from trace_operators import to_tensor
import pytorch.nn as nn
from functools import partial


class PartialTrace(ABC):
    """maybe just a DeepChainMap but that seems horribly imperative"""
    pass


class _ArrowsTo(PartialTrace):
    def __init__(self, *arg, **kwargs):
        """
        kwargs should be only tensors or distributions that have been observed
        under Trace machanisms, so we just use a trace here.
        """
        if len(arg) == 1 and isinstance(arg[0], Trace):
            self.trace = arg[0]
        elif len(arg) == 0:
            self.trace = Trace()
        else:
            raise TypeError("ArrowTo only takes a positional trace")

        for k, v in kwargs.items():
            self.trace[k] = v
            setattr(self, k, v)  # you are responsible for not using a reserved keyword.
        self.keys = frozenset(self.trace.keys())  # not a frozenset because we like pain

    def __str__(self):
        return "Partial({})".format(str(self.trace))

    def __repr__(self):
        return "Partial({})".format(repr(self.trace))

    def item_subset(self, keys):
        return [(k, v) for k, v in self.trace.items() if k not in keys]


def view(arr, keys: Union[List[str], str]):
    if not isinstance(keys, _ArrowsTo):
        raise TypeError("unexpected argument: {}".format(keys))

    if isinstance(keys, list):
        diff = arr.keys - keys
        if len(diff) != 0:
            raise TypeError("{} is not in trace: {}".format(diff, arr))
        return _ArrowsTo(**arr.item_subset(keys))

    elif isinstance(keys, str):
        return _ArrowsTo(**arr.item_subset([keys]))

    else:
        raise TypeError("unexpected argument: {}".format(keys))


def mempty():
    return _ArrowsTo()


def from_generator(g):
    return _ArrowsTo()


def mappend(left, right):
    if not (isinstance(left, PartialTrace) or isinstance(right, PartialTrace)):
        raise TypeError("can only add PartialTrace types")

    key_intersection = left.keys & right.keys
    if len(key_intersection) > 0:
        """ do this check both for the warning and because of kwarg collision """
        print("WARNING: overlapping keys found. mappend has right precidence.")
        # let the pain begin
        return _ArrowsTo(**left.trace.item_subset(key_intersection), **right.trace.items())
    else:
        return _ArrowsTo(**left.trace.items(), **right.trace.items())


class ArrowsTo(_ArrowsTo):
    """
    Awkward inheritance because python AFAICT doesn't let you reference a class
    inside that same class. Might be fixed in 3.8/3.9?

    Anyhow, I wonder if this should be called "a partition" -- technically all
    colliding arrows also form an equivalence class. not sure what to do with
    that information, yet.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        return self.trace

    def __add__(self, o):
        return mappend(self, o)


class Expr(object):
    pass


class LazyTrace(Expr):
    def __init__(self, trace: Union[Trace, ArrowsTo]):
        self.arrows = ArrowsTo(trace) if isinstance(trace, Trace) else trace

    def __add__(self, other):
        assert isinstance(other, self.__class__), "can only mappend {} instances".format(self.__class__)
        new_arrows = mappend(self.arrows, other.arrows)
        return LazyTrace(new_arrows)


class LazyDensity(Expr):
    def __init__(self, density):
        self.density = density
        self.thunk = None

    def apply(self, *args, **kwargs):
        # assumes that you can just keep appending to an appended function
        fn = self.density if self.thunk is None else self.thunk
        self.thunk = partial(fn, *args, **kwargs)
        return self

    def __call__(self):
        return (self.thunk if self.thunk is not None else self.density)()

    def __add__(self, other):
        assert isinstance(other, self.__class__), "can only + {} instances".format(self.__class__)
        t = Trace()
        t.append("_left", self())
        t.append("_right", other())
        return LazyTrace(t)





class ConditionedOnGenerator(Expr):
    def __init__(self, trace: LazyTrace, arrows: ArrowsTo):
        self.trace = trace
        self.arrows = arrows


class ConditionedOnValue(object):
    def __init__(self, trace):
        self.trace = trace
        self.tracekeys = set(trace.keys())

    def __call__(self):
        return self.trace

    # 1) No conditional trace, no value
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y
    #     ==> trace={'x': val_x, 'y': val_y~k,}
    # 2) No conditional trace, with value
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, val_y
    #     ==> trace={'x': val_x, 'y': val_y,}
    # 3) With conditional trace, no value.
    # (NOTE: Sam says that k should be a lens-like-thing parameterized by cond_trace)
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}
    #     ==> trace={'x': val_x, 'y': val_y',}
    # 4) With conditional trace, with value
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
    #     ==> ERROR: Unclear with val_y or val_y' should be used
    # 5) With conditional trace, with value
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
    #     ==> trace={'x': val_x, 'y': val_y',}
    #     Always prefer value over cond_trace!!! This is new behaviour see old behaviour below.
    # 5 - old behaviour) With conditional trace, with value
    # trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
    #     ==> ERROR: Unclear with val_y or val_y' should be used
    # """


l1 = _
l2 = _

lazy_condinition3 = l1 | l2 # <<< automatically condition and detach
condinition3 = run(lazy_condinition3)
# if isinstance(ConditionLazyTrace):
#     run_function_condition(conditoindlazy.run())

lazy_union = l1 & l2 # <<< automatically condition and detach


def lazy_extend1(lazy_trace: LazyTrace, gen, cond_map):
    detach_value = True
    cond_set = {cond_map[cond_name]: to_tensor(cond_trace[cond_name], detach_value) for cond_name in cond_map.keys()}

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=None, cond_set=cond_set, **kwargs)
    trace.append(node, name=name)
    return trace

def extend1(stochastic_gen, name, trace, cond_map=dict(), detach_value=False, **kwargs):
    """
    CASE: 1) No conditional trace, no value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y
        ==> trace={'x': val_x, 'y': val_y~k,}

    Extends an existing trace using a stochastic generator which can be conditioned on nodes in an
    optional conditional trace.
    name::string            name of the node that extends the trace
    trace::Trace            trace that is extended

    NOTE: If no conditional trace is passed the stochastic generator is conditioned on the target trace itself.
    NOTE: If no value is passed then it is taken from the conditional trace or otherwise generated by the stochastic
          generator. If the stochastic generator does not support the generation of values the user needs to make sure a
          value or a conditional trace which containing a suitable note is passed.
    NOTE: Sam says there might be a notion of subtyping for traces going on here
    """
    cond_trace = trace    # cond_trace::Trace       trace that the stochastic generator is conditioned on (NONE HERE)
    value = None          # value::Tensor           value of the node that extends the trace

    # if cond_map is not None:
    # cond_map::dict          maps node names to the corresponding names of the stochastic generator
    #
    # If cond_trace or cond_set is missing the cond_set for the stochastic generator will be None.
    cond_set = {cond_map[cond_name]: to_tensor(cond_trace[cond_name], detach_value) for cond_name in cond_map.keys()}

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=None, cond_set=cond_set, **kwargs)
    trace.append(node, name=name)
    return trace

def extend2(stochastic_gen, name, trace, value, cond_map=None, param_set={}, detach_value=False, **kwargs):
    """
    CASE: 2) No conditional trace, with value
        trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, val_y
            ==> trace={'x': val_x, 'y': val_y,}

    Extends an existing trace using a stochastic generator which can be conditioned on nodes in an
    optional conditional trace.
    name::string            name of the node that extends the trace
    value::Tensor           value of the node that extends the trace
    trace::Trace            trace that is extended
    cond_trace::Trace       trace that the stochastic generator is conditioned on
    cond_map::dict          maps node names to the corresponding names of the stochastic generator
    param_set::dict         additional parameter dependencies of the stochastic generator

    NOTE: If no conditional trace is passed the stochastic generator is conditioned on the target trace itself.
    NOTE: If no value is passed then it is taken from the conditional trace or otherwise generated by the stochastic
          generator. If the stochastic generator does not support the generation of values the user needs to make sure a
          value or a conditional trace which containing a suitable note is passed.
    NOTE: Sam says there might be a notion of subtyping for traces going on here

    """

    # if cond_trace is None:
    cond_trace = trace
    value = to_tensor(value, detach_value)  # <<< for the extend that is monadic bind, this is always true.

    # if cond_map is not None:
    # cond_map::dict          maps node names to the corresponding names of the stochastic generator
    #
    # If cond_trace or cond_set is missing the cond_set for the stochastic generator will be None.
    cond_set = {cond_map[cond_name]: to_tensor(cond_trace[cond_name], detach_value) for cond_name in cond_map.keys()}

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=value, cond_set=cond_set, param_set=param_set, **kwargs)
    trace.append(node, name=name)
    return trace


def extend3(stochastic_gen, name, trace, cond_trace, cond_map=None, detach_value=False, **kwargs):
    """
    CASE 3) With conditional trace, no value.
         (NOTE: Sam says that k should be a lens-like-thing parameterized by cond_trace)
         trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}
             ==> trace={'x': val_x, 'y': val_y',}

    Extends an existing trace using a stochastic generator which can be conditioned on nodes in an
    optional conditional trace.
    name::string            name of the node that extends the trace
    value::Tensor           value of the node that extends the trace
    trace::Trace            trace that is extended
    cond_trace::Trace       trace that the stochastic generator is conditioned on
    cond_map::dict          maps node names to the corresponding names of the stochastic generator

    NOTE: If no conditional trace is passed the stochastic generator is conditioned on the target trace itself.
    NOTE: If no value is passed then it is taken from the conditional trace or otherwise generated by the stochastic
          generator. If the stochastic generator does not support the generation of values the user needs to make sure a
          value or a conditional trace which containing a suitable note is passed.
    NOTE: Sam says there might be a notion of subtyping for traces going on here
    """
    if name in cond_trace:
        value = cond_trace[name]
    else:
        value = None

    # Build conditioning set
    cond_set = {}
    if cond_trace is not None and cond_map is not None:
        # If cond_trace or cond_set is missing the cond_set for the stochastic generator will be None.
        for cond_name in cond_map.keys():
            cond_set[cond_map[cond_name]] = to_tensor(cond_trace[cond_name], detach_value)

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=value, cond_set=cond_set, param_set=param_set, **kwargs)
    trace.append(node, name=name)
    return trace

def extend4(stochastic_gen, name, trace, value=None, cond_trace=None, cond_map=None, detach_value=False, **kwargs):
    """
    CASE 3) With conditional trace, no value.
         (NOTE: Sam says that k should be a lens-like-thing parameterized by cond_trace)
         trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}
             ==> trace={'x': val_x, 'y': val_y',}

    CASE 4) With conditional trace, with value
         trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
            ==> ERROR: Unclear with val_y or val_y' should be used

    Extends an existing trace using a stochastic generator which can be conditioned on nodes in an
    optional conditional trace.
    name::string            name of the node that extends the trace
    value::Tensor           value of the node that extends the trace
    trace::Trace            trace that is extended
    cond_trace::Trace       trace that the stochastic generator is conditioned on
    cond_map::dict          maps node names to the corresponding names of the stochastic generator

    NOTE: If no conditional trace is passed the stochastic generator is conditioned on the target trace itself.
    NOTE: If no value is passed then it is taken from the conditional trace or otherwise generated by the stochastic
          generator. If the stochastic generator does not support the generation of values the user needs to make sure a
          value or a conditional trace which containing a suitable note is passed.
    NOTE: Sam says there might be a notion of subtyping for traces going on here
    """
    if cond_trace is None:
        cond_trace = trace
    else:
        # TODO: check enabling of valid secenario 4
        if value is not None:
            # raise ValueError("Scenario 4")
            pass
        elif name in cond_trace:
            value = cond_trace[name]
    value = None if value is None else to_tensor(value, detach_value)  # <<< for the extend that is monadic bind, this is always true.

    # Build conditioning set
    cond_set = {}
    if cond_trace is not None and cond_map is not None:
        # If cond_trace or cond_set is missing the cond_set for the stochastic generator will be None.
        for cond_name in cond_map.keys():
            cond_set[cond_map[cond_name]] = to_tensor(cond_trace[cond_name], detach_value)

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=value, cond_set=cond_set, param_set=param_set, **kwargs)
    trace.append(node, name=name)
    return trace

# ======================================================
#
#
def mk_singleton_trace(stochastic_gen):
    # Get stochastic node and append it to trace
    trace = Trace()
    node = stochastic_gen(value=value, cond_set=cond_set, param_set=param_set, **kwargs)
    trace.append(node, name=name)
    return trace


trace1 <- Trace()
trace2 <- mk_singleton_trace(stochastic_gen)

#
trace1 + trace2 # append
trace1 | trace2 # condition trace1 on trace2
#        ------
#          |
#           \_  trace2 > trace1
#               trace2 < trace1
#               trace2 = trace1

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


def extend(stochastic_gen, name, trace, value=None, cond_trace=None, cond_map=None, param_set={},
           detach_value=False, **kwargs):
    """
    Extends an existing trace using a stochastic generator which can be conditioned on nodes in an
    optional conditional trace.
    name::string            name of the node that extends the trace
    value::Tensor           value of the node that extends the trace
    trace::Trace            trace that is extended
    cond_trace::Trace       trace that the stochastic generator is conditioned on
    cond_map::dict          maps node names to the corresponding names of the stochastic generator
    param_set::dict         additional parameter dependencies of the stochastic generator

    NOTE: If no conditional trace is passed the stochastic generator is conditioned on the target trace itself.
    NOTE: If no value is passed then it is taken from the conditional trace or otherwise generated by the stochastic
          generator. If the stochastic generator does not support the generation of values the user needs to make sure a
          value or a conditional trace which containing a suitable note is passed.
    NOTE: Sam says there might be a notion of subtyping for traces going on here

    base address => by lexical src
    + count => number of calls

    fresh-vars: counts _can_ collide on a per-trace situation (think of encoder-decoder)

    Toy scenarios:
    1) No conditional trace, no value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y
        ==> trace={'x': val_x, 'y': val_y~k,}

    2) No conditional trace, with value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, val_y
        ==> trace={'x': val_x, 'y': val_y,}

    3) With conditional trace, no value.
    (NOTE: Sam says that k should be a lens-like-thing parameterized by cond_trace)
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}
        ==> trace={'x': val_x, 'y': val_y',}

    Trace {a: Tensor} >>= f :: ({a: Tensor} -> Trace _) >>=

    4) With conditional trace, with value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
        ==> ERROR: Unclear with val_y or val_y' should be used

    5) With conditional trace, with value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
        ==> trace={'x': val_x, 'y': val_y',}
        Always prefer value over cond_trace!!! This is new behaviour see old behaviour below.

    5 - old behaviour) With conditional trace, with value
    trace={'x': val_x,}; cond_map:'x'->'x_in'; k: val_x->val_y, x|->y, cond_trace={'x': val_x', 'y': val_y',}, val_y
        ==> ERROR: Unclear with val_y or val_y' should be used
    """
    if cond_trace is None:
        cond_trace = trace
    else:
        # TODO: check enabling of valid secenario 4
        if value is not None:
            # raise ValueError("Scenario 4")
            pass
        elif name in cond_trace:
            value = cond_trace[name]
    value = None if value is None else to_tensor(value, detach_value)  # <<< for the extend that is monadic bind, this is always true.

    # Build conditioning set
    cond_set = {}
    if cond_trace is not None and cond_map is not None:
        # If cond_trace or cond_set is missing the cond_set for the stochastic generator will be None.
        for cond_name in cond_map.keys():
            cond_set[cond_map[cond_name]] = to_tensor(cond_trace[cond_name], detach_value)

    # Get stochastic node and append it to trace
    node = stochastic_gen(value=value, cond_set=cond_set, param_set=param_set, **kwargs)
    trace.append(node, name=name)
    return trace
















































































_ = """
Simpler version from nvi.probtorch.generator

    def extend(self, name, trace, value=None, cond_trace=None, cond_map={}, param_set={},
               detach_value=False, detach_parameters=False, **kwargs):
        if cond_trace is None:
            cond_trace = trace
        # If there is a cond. trace, check if new rv is observed, i.e. rv already exists in cond. trace
        elif name in cond_trace:
            value = cond_trace[name]

        if value is not None:
            value = to_tensor(value, detach_value)

        # Build conditioning set from rvs in cond. trace or from trace
        cond_set = {}
        for var_name_gen, var_name in cond_map.items():
            cond_set[var_name_gen] = to_tensor(cond_trace[var_name], detach_value)

        # Get stochastic nodes and append them to the trace
        node = self.forward(value=value, cond_set=cond_set, param_set=param_set,
                            detach_parameters=detach_parameters, **kwargs)
        node.generator = self
        trace.append(node, name=name)
        assert not (detach_value and not (node.value.grad_fn is None))
        return trace

"""
