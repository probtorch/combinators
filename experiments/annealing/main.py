#!/usr/bin/env python3
import torch
import math
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple

import combinators.trace.utils as trace_utils
from combinators.densities import MultivariateNormal, Tempered, RingGMM
from combinators.densities.kernels import MultivariateNormalKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl
from combinators import Forward, Reverse, Propose
from combinators.stochastic import RandomVariable, ImproperRandomVariable

def mk_kernel(target:int, std:float, num_hidden:int):
    embedding_dim = 2
    return MultivariateNormalKernel(
        ext_name=f'g_{target}',
        loc=torch.zeros(2),
        cov=torch.eye(2)*std**2,
        net=ResMLPJ(
            dim_in=2,
            dim_hidden=num_hidden,
            dim_out=embedding_dim,
        ))

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


def main(steps=8, lr=1e-3, num_iterations=300, num_samples=200):
    K = steps
    torch.manual_seed(1)
    out = mk_model(K)
    targets, forwards, reverses = [out[n] for n in ['targets', 'forwards', 'reverses']]
    assert all([len(list(t.parameters())) == 0 for t in targets])
    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = num_iterations
    sample_shape=(num_samples,)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)
            loss = torch.zeros(1)

            lvs = []
            lw = torch.zeros(sample_shape)

            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)
                # FIXME: because p_prv_tr is not eliminating the previous trace, the trace is cumulativee but removing grads leaves backprop unaffected
                p_prv_tr = state.target.trace
                q.clear_observations()
                p.clear_observations()

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g_{k}'], state.target.trace[f'g_{k+1}'])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0

    with torch.no_grad():
        tol = Tolerance(loc=0.15, scale=0.15)
        print("made it to test phase!")


if __name__ == '__main__':
    main(num_iterations=300)
    print('type-checks!')
