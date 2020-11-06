#!/usr/bin/env python3

class LazyTrace(object):
    def __init__(self, trace):
        self.trace = trace

    def __call__(self):
        return self.trace

def dot(a, b):
    return None

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
    value = None if value is None else to_tensor(value, detach_value)

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
