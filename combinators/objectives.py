#!/usr/bin/env python
import torch
import math
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from combinators.trace import utils as trace_utils
import combinators.stochastic as probtorch
from combinators.stochastic import RandomVariable, ImproperRandomVariable

def _estimate_mc(values: Tensor, log_weights: Tensor, sample_dims: Tuple[int], reducedims: Tuple[int], keepdims: bool) -> Tensor:
    if len(log_weights.shape) == 1:
        return values.sum(dim=reducedims, keepdim=keepdims)
    else:
        nw = F.softmax(log_weights, dim=sample_dims)
        return (nw * values).sum(dim=reducedims, keepdim=keepdims)

def nvo_avo(out, sample_dims=0) -> Tensor:
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

    lw = out.log_weight.detach()
    lv = out.lv
    proposal_trace = out.proposal_trace
    target_trace = out.target_trace

    if reducedims is None:
        reducedims = (sample_dims,)
    rv_proposal = list(proposal_trace.values())[1]
    rv_target = list(target_trace.values())[0]

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

    ## Fully reparameterized gradient estimator
    if isinstance(rv_proposal, RandomVariable) and rv_proposal.reparameterized:
        # Compute reparameterized gradient
        kl_term = _estimate_mc(f,
                               lw,
                               sample_dims=sample_dims,
                               reducedims=reducedims,
                               keepdims=False,
                               )
        return kl_term - _eval0(grad_log_Z1_term) + _eval0(grad_log_Z2_term)

    ## Score function gradient gradient estimator
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
