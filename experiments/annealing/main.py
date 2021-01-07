#!/usr/bin/env python3
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from experiments.annealing.dataset import RingGMM, MultivariateNormalKernel
from combinators.densities import MultivariateNormal, Tempered

def mk_kernel(target:int, std:float, num_hidden:int):
    return MultivariateNormalKernel(ext_name=f'g_{target}', loc=torch.zeros(2), cov=torch.eye(2)*std**2, dim_hidden=num_hidden)

def mk_model(num_targets:int):
    proposal_std = 1.0
    g_0 = MultivariateNormal(name='g_0', loc=torch.zeros(2), cov=torch.eye(2)*proposal_std**2)
    g_K = RingGMM(scale=8, count=8, name=f"g_{num_targets}")

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

def main(steps=8):
    out = mk_model(steps)

if __name__ == '__main__':
    main()
