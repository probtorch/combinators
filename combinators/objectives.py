#!/usr/bin/env python
import torch
import math
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from combinators.trace import utils as trace_utils
import combinators.stochastic as probtorch
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.trace.utils import valeq

def _estimate_mc(values: Tensor, log_weights: Tensor, sample_dims: Tuple[int], reducedims: Tuple[int], keepdims: bool) -> Tensor:
    nw = F.softmax(log_weights, dim=sample_dims)
    return (nw * values).sum(dim=reducedims, keepdim=keepdims)

def nvo_avo(out, sample_dims=0) -> Tensor:
    # proposal_trace = out.proposal_trace
    # target_trace = out.target_trace
    # rv_proposal = list(proposal_trace.values())[1]
    # rv_target = list(target_trace.values())[0]

    f = -out.lv
    log_weights = torch.zeros_like(out.lv)

    nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    loss = (nw * f).sum(dim=(sample_dims,), keepdim=False)
    return loss

def nvo_rkl(
    out,
    sample_dims=0,
    reducedims=None,
    **kwargs,
) -> Tensor:
    if reducedims is None:
        reducedims = (sample_dims,)

    lw = out.log_weight.detach()
    lv = out.lv
    proposal_trace = out.proposal_trace
    target_trace = out.target_trace
    rv_proposal_name, rv_proposal = list(proposal_trace.items())[1]
    rv_target_name, rv_target = list(target_trace.items())[0]

    lw = lw.detach()
    ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
    f = -(lv - ldZ)
    grad_log_Z1_term = _estimate_mc(rv_proposal._log_prob,
                                    lw,
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_proposal, probtorch.RandomVariable) else torch.tensor(0.)
    grad_log_Z2_term = _estimate_mc(_eval_nrep(rv_target)._log_prob,
                                    lw+lv.detach(),
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_target, probtorch.RandomVariable) else torch.tensor(0.)

    ## Score function gradient gradient estimator
    assert not (isinstance(rv_proposal, RandomVariable) and rv_proposal.reparameterized)
    baseline = _estimate_mc(f.detach(),
                            lw,
                            sample_dims=sample_dims,
                            reducedims=reducedims,
                            keepdims=False,
                            )
    kl_term = _estimate_mc((f - baseline) * _eval1(rv_proposal._log_prob - grad_log_Z1_term),
                           lw,
                           sample_dims=sample_dims,
                           reducedims=reducedims,
                           keepdims=False,
                           )
    loss = kl_term + _eval0(grad_log_Z2_term) + baseline

    # Tests
    rv_fwd_name, rv_fwd= list(proposal_trace.items())[0]
    rv_rev_name, rv_rev= list(target_trace.items())[0]
    assert(len(proposal_trace) == 2)
    assert(len(target_trace) == 2)
    assert valeq(proposal_trace, target_trace)
    assert (torch.equal(target_trace.log_joint(sample_dims=sample_dims, batch_dim=1) - proposal_trace.log_joint(sample_dims=sample_dims, batch_dim=1),
                        lv))
    assert (rv_proposal.log_prob.grad_fn is None), "will fail for *"
    assert (rv_fwd.log_prob.grad_fn is not None)
    assert (rv_target.log_prob.grad_fn is not None)
    assert (_eval_nrep(rv_target)._log_prob.grad_fn is None), "will fail for star"
    assert (rv_rev.log_prob.grad_fn is not None)
    return loss

def _eval0(e):
    return e - e.detach()

def _eval1(e):
    return torch.exp(_eval0(e))

def _eval_nrep(rv):
    value = rv.value.detach()
    if isinstance(rv, RandomVariable):
        return RandomVariable(value=value, dist=rv.dist, provenance=rv.provenance, reparameterized=rv.reparameterized)
    elif isinstance(rv, ImproperRandomVariable):
        return ImproperRandomVariable(value=value, log_density_fn=rv.log_density_fn, provenance=rv.provenance)
    else:
        raise NotImplementedError("Only supports RandomVariable and ImproperRandomVariable")
