#!/usr/bin/env python
import torch
import math
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from combinators.trace import utils as trace_utils
import combinators.stochastic as probtorch

def _estimate_mc(values: Tensor, log_weights: Tensor, sample_dims: Tuple[int], reducedims: Tuple[int], keepdims: bool) -> Tensor:
    if len(log_weights.shape) == 1:
        return values.sum(dim=reducedims, keepdim=keepdims)
    else:
        nw = F.softmax(log_weights, dim=sample_dims)
        return (nw * values).sum(dim=reducedims, keepdim=keepdims)

def nvo_avo(lv: Tensor, sample_dims=0) -> Tensor:
    # values = -lv
    # log_weights = torch.zeros_like(lv)

    # nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    # loss = (nw * values).sum(dim=(sample_dims,), keepdim=False)
    loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
    return loss

def mb0(e):
    return e - e.detach()

def mb1(e):
    return torch.exp(mb0(e))
_eval0 = mb0 # magicbox 0
_eval1 = mb1 # magicbox 1
def eval_nrep(rv):
    return trace_utils.copyrv(rv, requires_grad=False)

def nvo_rkl(lw: Tensor, lv: Tensor, rv_proposal, rv_target, batch_dim=None, sample_dims=0) -> Tensor:
    # TODO: move back from the proposal and target RVs to joint logprobs?
    reducedims = (sample_dims,)

    lw = lw.detach()
    ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
    f = -lv

    # rv_proposal = next(iter(proposal_trace.values())) # tr[\gamma_{k-1}]
    # rv_target = next(iter(target_trace.values()))     # tr[\gamma_{k}]

    kwargs = dict(
        sample_dims=sample_dims,
        reducedims=reducedims,
        keepdims=False
    )

    baseline = _estimate_mc(f.detach(), lw, **kwargs).detach()

    kl_term = _estimate_mc(mb1(rv_proposal._log_prob) * (f - baseline), lw, **kwargs)

    grad_log_Z1 = _estimate_mc(rv_proposal._log_prob, lw, **kwargs)
    grad_log_Z2 = _estimate_mc(eval_nrep(rv_target)._log_prob, lw+lv.detach(), **kwargs)

    loss = kl_term + mb0(baseline * grad_log_Z1 - grad_log_Z2) + baseline + ldZ
    return loss

def _nvo_rkl(
    lw: Tensor,  # Log cumulative weight, this is not detached yet - see if it bites Babak and Hao
    lv: Tensor,
    rv_proposal, rv_target,

    batch_dim=None,
    sample_dims=0,
    reducedims=None,
) -> Tensor:
    if reducedims is None:
        reducedims = (sample_dims,)
    # rv_proposal = _unpack(proposal_trace, 0)
    # rv_target = _unpack(target_trace, 0)
    lw = lw.detach()
    ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
    f = -(lv - ldZ)
    grad_log_Z1_term = _estimate_mc(rv_proposal._log_prob,
                                    lw,
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_proposal, probtorch.RandomVariable) else torch.tensor(0.)
    grad_log_Z2_term = _estimate_mc(eval_nrep(rv_target)._log_prob,
                                    lw+lv.detach(),
                                    sample_dims=sample_dims,
                                    reducedims=reducedims,
                                    keepdims=False,
                                    ) if not isinstance(rv_target, probtorch.RandomVariable) else torch.tensor(0.)
    # if rv_proposal.generator.reparameterized:
    #     # Compute reparameterized gradient
    #     kl_term = _estimate_mc(f,
    #                            lw,
    #                            sample_dims=sample_dims,
    #                            reducedims=reducedims,
    #                            keepdims=False,
    #                            )
    #     return kl_term - _eval0(grad_log_Z1_term) + _eval0(grad_log_Z2_term)
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

def nvo_rkl_1d(lw: Tensor, lv: Tensor, rv_proposal, rv_target, batch_dim=None, sample_dims=0) -> Tensor:
    # TODO: move back from the proposal and target RVs to joint logprobs?
    reducedims = (sample_dims,)

    lw = lw.detach()
    ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
    f = -lv

    # rv_proposal = next(iter(proposal_trace.values())) # tr[\gamma_{k-1}]
    # rv_target = next(iter(target_trace.values()))     # tr[\gamma_{k}]

    kwargs = dict(
        sample_dims=sample_dims,
        reducedims=reducedims,
        keepdims=False
    )

    baseline = _estimate_mc(f.detach(), lw, **kwargs).detach()

    kl_term = _estimate_mc(mb1(rv_proposal.log_prob.squeeze()) * (f - baseline), lw, **kwargs)

    grad_log_Z1 = _estimate_mc(rv_proposal.log_prob.squeeze(), lw, **kwargs)
    grad_log_Z2 = _estimate_mc(eval_nrep(rv_target).log_prob.squeeze(), lw+lv.detach(), **kwargs)

    loss = kl_term + mb0(baseline * grad_log_Z1 - grad_log_Z2) + baseline + ldZ
    return loss
