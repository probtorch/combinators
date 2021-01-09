#!/usr/bin/env python3
import torch
import math
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from experiments.annealing.dataset import RingGMM, MultivariateNormalKernel
from combinators.densities import MultivariateNormal, Tempered
from combinators import Forward, Reverse, Propose
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from tqdm import trange
from typing import Tuple

def mk_kernel(target:int, std:float, num_hidden:int):
    return MultivariateNormalKernel(ext_name=f'g_{target}', loc=torch.zeros(2), cov=torch.eye(2)*std**2, dim_hidden=num_hidden)

def mk_model(num_targets:int):
    proposal_std = 1.0
    g_0 = MultivariateNormal(name='g_0', loc=torch.zeros(2), cov=torch.eye(2)*proposal_std**2)
    g_K = RingGMM(scale=8, count=8, name=f"g_{num_targets}")

    # Make an annealing path
    betas = torch.arange(0., 1., 1./(num_targets - 1))[1:] # g_0 is beta=0
    path = [Tempered(f'g_{k}', g_0, g_K, beta) for k, beta in enumerate(betas)]
    path = [g_0] + path + [g_K]
    assert len(path) == num_targets # sanity check that the betas line up

    num_kernels = num_targets - 1
    target_ixs = [ix for ix in range(0, num_targets)]
    mk_kernels = lambda shift_target: [
        mk_kernel(target=shift_target(ix),
                  std=shift_target(ix)+1.0,
                  num_hidden=64
                  ) for ix in target_ixs[:num_kernels]
    ]

    return dict(
        targets=path,
        forwards=mk_kernels(lambda ix: ix+1),
        reverses=mk_kernels(lambda ix: ix),
    )

def _estimate_mc(values: Tensor, log_weights: Tensor, sample_dims: Tuple[int], reducedims: Tuple[int], keepdims: bool) -> Tensor:
    nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    return (nw * values).sum(dim=reducedims, keepdim=keepdims)

def mb1(e):
    return torch.exp(e - e.detach())

def mb0(e):
    return e - e.detach()

def eval_nrep(rv):
    if isinstance(rv, RandomVariable):
        return RandomVariable(rv.dist, rv.value.detach())
    elif isinstance(rv, ImproperRandomVariable):
        return ImproperRandomVariable(rv.log_density_fn, rv.value.detach())
    else:
        raise ValueError("Node type not supported")

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


def main(steps=8, lr=1e-3, num_iterations=300):
    K = steps
    out = mk_model(K)
    targets, forwards, reverses = [out[n] for n in ['targets', 'forwards', 'reverses']]
    assert all([len(list(t.parameters())) == 0 for t in targets])
    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])
    sample_shape=(200,)

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=lr)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_iterations) as bar:
        for i in bar:
            q0 = targets[0]
            q_prv_tr, out0 = q0(sample_shape=sample_shape)
            loss = torch.zeros([1])
            lw = torch.zeros_like(out0)

            for fwd, rev, q, p, k_tar in zip(forwards, reverses, targets[:-1], targets[1:], range(1,K+1)):
                q.condition_on(q_prv_tr)
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state, lv = extend(sample_shape=sample_shape)
                q_prv_tr = state.proposal.trace

                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g_{k}'], state.target.trace[f'g_{k}'])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if num_steps <= 100:
               loss_avgs.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_avgs.append(loss_avg)
    with torch.no_grad():
        report_sparkline(loss_avgs)

if __name__ == '__main__':
    main(num_iterations=300)
    print('type-checks!')
