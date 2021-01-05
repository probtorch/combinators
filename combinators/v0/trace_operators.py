import torch
import probtorch

from typing import *
from typeguard import typechecked
from probtorch import ImproperRandomVariable, RandomVariable, Trace

import combinators.resampling as rs
from combinators.normals import extend as extend_normal

@typechecked
def to_tensor(val: Union[float, list, Tensor, probtorch.RandomVariable, probtorch.ImproperRandomVariable], detach=False):
    if isinstance(val, float):
        val = torch.tensor(val)
    if isinstance(val, list):
        val = tensor(val)
    if isinstance(val, Tensor):
        val = val
    if isinstance(val, probtorch.RandomVariable) or isinstance(val, probtorch.ImproperRandomVariable):
        val = val.value
    if detach:
        val = val.detach()
    return val

# ---------------------
# Basic trace operations
# ---------------------

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


def trace(stochastic_gen, name, value=None, cond_trace=None, cond_map=None, param_set={},
          detach_value=False, **kwargs):
    """
    Generates a new trace containing a node generated by the stochastic generator.
    -> See extend
    """
    return extend(stochastic_gen, name, Trace(), value=value,
                  cond_trace=cond_trace, cond_map=cond_map, param_set=param_set,
                  detach_value=detach_value, **kwargs)


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



# --------------------------
#  Inferece trace operations
# --------------------------

def log_importance_weight(proposal_trace, target_trace, batch_dim=1, sample_dims=0, check_exist=True):
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

def resample(trace, log_weight, batch_dim=1, sample_dims=0):
    aidx = rs.ancestor_indices_systematic(log_weight, sample_dim=sample_dims, batch_dim=batch_dim)
    new_trace = Trace()
    for key in trace._nodes:
        # TODO: Do not just detach
        value = rs.pick(trace[key].value, aidx, sample_dim=sample_dims).detach()
        log_prob = rs.pick(trace[key].log_prob, aidx, sample_dim=sample_dims).detach()
        if isinstance(trace[key], RandomVariable):
            var = RandomVariable(dist=trace[key]._dist, value=value, log_prob=log_prob)
        elif isinstance(trace[key], ImproperRandomVariable):
            var = ImproperRandomVariable(log_density_fn=trace[key].log_density_fn, value=value, log_prob=log_prob)
        new_trace.append(var, name=key)
    log_weight = torch.logsumexp(log_weight - log_weight.shape[sample_dims], dim=sample_dims, keepdim=True).expand_as(log_weight)
    return new_trace, log_weight

def resample_(trace, log_weight, batch_dim=1, sample_dims=0):
    aidx = rs.ancestor_indices_systematic(log_weight, sample_dim=sample_dims, batch_dim=batch_dim)
    for key in trace._nodes:
        trace[key]._value = rs.pick(trace[key].value, aidx, sample_dim=sample_dims)
        trace[key]._log_prob = rs.pick(trace[key].log_prob, aidx, sample_dim=sample_dims)
    log_weight = torch.logsumexp(log_weight, dim=sample_dims, keepdim=True).expand_as(log_weight)
    return trace, log_weight

# ---------------------------
# Proper weighting operations
# ---------------------------

def properly_weight(pw_op, target_trace):
    """
    Applies a proper weighting operation to the target trace.
    Applying a proper weighting operation results in a new log weight and a - possibly modified - target trace which act
    as a properly weighted pair for the desity defined by the target trace.
    """
    return pw_op(target_trace)

def importance_weighting(proposal_trace, lw=0., batch_dim=1, sample_dims=0, check_exist=True):
    """
    Compute importance weight of the proposal trace under the target trace considering the incomming log weight lw of
    the proposal trace
    """
    def pw_op(target_trace):
        lv = log_importance_weight(proposal_trace, target_trace, batch_dim=batch_dim, sample_dims=sample_dims,
                                   check_exist=check_exist)
        return target_trace, lw + lv
    return pw_op

def importance_resampling(proposal_trace, lw=0., batch_dim=1, sample_dims=0):
    """
    Compute importance weight of the proposal trace under the target trace considering the incomming log weight lw of
    the proposal trace
    """
    def pw_op(target_trace):
        lv = log_importance_weight(proposal_trace, target_trace, batch_dim=batch_dim, sample_dims=sample_dims)
        target_trace, lw_ = resample(target_trace, lw + lv)
        return target_trace, lw_
    return pw_op

# --------------------
# Trace Transformation
# --------------------

def merge(name_merged, names, trace_ext, vdist, kernel, reverse_order=False):
    if reverse_order:
        names.reverse()
    if len(names) > 2:
        raise ValueError('Closed form analytic extend is currently only supported a two variables at a time')
    if not isinstance(trace_ext[names[0]].dist, torch.distributions.MultivariateNormal):
        raise ValueError('Closed form analytic extend is currently only supported for MultivariateNormalNormal')
    value_ext = torch.cat((to_tensor(trace_ext[names[0]]), to_tensor(trace_ext[names[1]])), dim=-1)
    mnormal = trace_ext[names[0]].dist

    # Analytically compute the joint
    B = kernel.cov_embedding.unembed(getattr(kernel, kernel.cov_embedding.embed_name), kernel.dim)
    F, t = kernel.map.weight, kernel.map.bias
    normal_ext = extend_normal(mnormal, F, t, B, reverse_order=reverse_order)
    # Add to trace
    provenance = probtorch.Provenance.OBSERVED if value_ext is None else probtorch.Provenance.SAMPLED
    trace_ext_analytic = probtorch.Trace()
    trace_ext_analytic.append(probtorch.RandomVariable(normal_ext, value_ext, provenance=provenance), name=name_merged)
    return trace_ext_analytic