#!/usr/bin/env python3
import torch
import math
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt

import combinators.trace.utils as trace_utils
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.densities import MultivariateNormal, Tempered, RingGMM
from combinators.densities.kernels import MultivariateNormalKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl
from combinators import Forward, Reverse, Propose
from combinators.stochastic import RandomVariable, ImproperRandomVariable


def mk_kernel(from_:int, to_:int, std:float, num_hidden:int):
    embedding_dim = 2
    return MultivariateNormalKernel(
        ext_from=f'g{from_}',
        ext_to=f'g{to_}',
        loc=torch.zeros(2, **kw_autodevice()),
        cov=torch.eye(2, **kw_autodevice())*std**2,
        net=ResMLPJ(
            dim_in=2,
            dim_hidden=num_hidden,
            dim_out=embedding_dim).to(autodevice()))

def mk_model(num_targets:int):
    proposal_std = 1.0
    g0 = MultivariateNormal(name='g0', loc=torch.zeros(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())*proposal_std**2)
    gK = RingGMM(scale=8, count=8, name=f"g{num_targets - 1}").to(autodevice())

    # Make an annealing path
    betas = torch.arange(0., 1., 1./(num_targets - 1))[1:] # g_0 is beta=0
    path = [Tempered(f'g{k}', g0, gK, beta) for k, beta in zip(range(1,num_targets-1), betas)]
    path = [g0] + path + [gK]
    assert len(path) == num_targets # sanity check that the betas line up

    return dict(
        targets=path,
        forwards= [mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets)],
        reverses= [mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets)],
    )


def main(seed=1, num_iterations=10000):
    # Setup
    torch.manual_seed(seed)
    K = 8
    num_samples = 200
    sample_shape=(num_samples,)

    # Models
    out = mk_model(K)
    targets, forwards, reverses = [out[n] for n in ['targets', 'forwards', 'reverses']]
    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-4)

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_iterations) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros(1, **kw_autodevice())
            lw, lvs = torch.zeros(sample_shape, **kw_autodevice()), []
            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q, _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)

                p_prv_tr = state.target.trace
                p.clear_observations()
                q.clear_observations()

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
                lvs.append(lv)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            writer.add_scalar('Loss/train', loss_scalar, i)
            loss_sum += loss_scalar
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               loss_ct, loss_sum  = 0, 0.0
               bar.set_postfix_str(loss_template)


    with torch.no_grad():
        breakpoint();

        tol = Tolerance(loc=0.15, scale=0.15)
        print("made it to test phase!")


if __name__ == '__main__':
    main()
    print('type-checks!')
