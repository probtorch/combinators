#!/usr/bin/env python3
import torch
import math
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Tuple
from matplotlib import pyplot as plt

import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
from combinators.densities.kernels import ConditionalMultivariateNormal, ConditionalMultivariateNormalLinear
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl, nvo_avo
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat

def mk_kernel(from_:int, to_:int, std:float, num_hidden:int, learn_cov=True, activation=nn.ReLU):
    embedding_dim = 2
    return ConditionalMultivariateNormal(
        ext_from=f'g{from_}',
        ext_to=f'g{to_}',
        loc=torch.zeros(2, **kw_autodevice()),
        cov=torch.eye(2, **kw_autodevice())*std**2,
        learn_cov=learn_cov,
        net=ResMLPJ(
            dim_in=2,
            dim_hidden=num_hidden,
            activation=activation,
            dim_out=embedding_dim).to(autodevice()))

def mk_mnlinear_kernel(from_:int, to_:int, std:float, dim:int):
    return MultivariateNormalLinearKernel(
        ext_from=f'g{from_}',
        ext_to=f'g{to_}',
        loc=torch.zeros(dim, **kw_autodevice()),
        cov=torch.eye(dim, **kw_autodevice())*std**2)

def mk_nlinear_kernel(from_:int, to_:int, std:float, dim:int):
    return NormalLinearKernel(ext_from=f'g{from_}', ext_to=f'g{to_}')

def anneal_to_ring(num_targets):
    proposal_std = 5
    g0 = mk_mvn(0, 0, std=proposal_std)
    gK = RingGMM(loc_scale=3, scale=0.16, count=2, name=f"g{num_targets - 1}").to(autodevice())
    return anneal_between(g0, gK, num_targets)

def anneal_between(left, right, total_num_targets):
    proposal_std = total_num_targets

    # Make an annealing path
    betas = torch.arange(0., 1., 1./(total_num_targets - 1))[1:] # g_0 is beta=0
    path = [Tempered(f'g{k}', left, right, beta) for k, beta in zip(range(1,total_num_targets-1), betas)]
    path = [left] + path + [right]

    assert len(path) == total_num_targets # sanity check that the betas line up
    return path


def anneal_between_mvns(left_loc, right_loc, total_num_targets):
    g0 = mk_mvn(0, left_loc)
    gK =  mk_mvn(total_num_targets-1, right_loc)

    return anneal_between(g0, gK, total_num_targets)

def anneal_between_ns(left_loc, right_loc, total_num_targets):
    g0 = mk_n(0, left_loc)
    gK =  mk_n(total_num_targets-1, right_loc)

    return anneal_between(g0, gK, total_num_targets)

def mk_mvn(i, loc, std=1):
    return MultivariateNormal(name=f'g{i}', loc=torch.ones(2, **kw_autodevice())*loc, cov=torch.eye(2, **kw_autodevice())*std**2)

def mk_n(i, loc):
    return Normal(name=f'g{i}', loc=torch.ones(1, **kw_autodevice())*loc, scale=torch.ones(1, **kw_autodevice())**2)

def paper_model():
    num_targets = 8
    g0 = mk_mvn(0, 0, std=5)
    gK = RingGMM(loc_scale=10, scale=0.5, count=8, name=f"g{num_targets - 1}").to(autodevice())

    def paper_kernel(from_:int, to_:int, std:float):
        return mk_kernel(from_, to_, std, num_hidden=50, learn_cov=True, activation=nn.Sigmoid)

    return dict(
        targets=anneal_between(g0, gK, num_targets),
        forwards=[paper_kernel(from_=i, to_=i+1, std=1.) for i in range(num_targets-1)],
        reverses=[paper_kernel(from_=i+1, to_=i, std=1.) for i in range(num_targets-1)],
    )

def mk_model(num_targets:int):
    return dict(
        targets=anneal_to_ring(num_targets),
        forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets-1)],
        reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets-1)],

#         targets=anneal_between_mvns(0, num_targets*2, num_targets),
#         forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets-1)],
#         reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets-1)],

#         targets=anneal_between_mvns(0, num_targets*2, num_targets),
#         forwards=[mk_mnlinear_kernel(from_=i, to_=i+1, std=1., dim=2) for i in range(num_targets-1)],
#         reverses=[mk_mnlinear_kernel(from_=i+1, to_=i, std=1., dim=2) for i in range(num_targets-1)],

        # NOTES: Anneal between 2 1d guassians with a linear kernel: 2 steps
        # annealing does not learn the forward kernel in the first step, but learns both in the second step.
#         targets=anneal_between_ns(0, num_targets*2, num_targets),
#         forwards=[mk_nlinear_kernel(from_=i, to_=i+1, std=1., dim=1) for i in range(num_targets-1)],
#         reverses=[mk_nlinear_kernel(from_=i+1, to_=i, std=1., dim=1) for i in range(num_targets-1)],

#         targets=[mk_mvn(i, i*2) for i in range(num_targets)],
#         forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=32) for i in range(num_targets-1)],
#         reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=32) for i in range(num_targets-1)],

#         targets=[mk_mvn(i, i*2) for i in range(num_targets)],
#         forwards=[mk_mnlinear_kernel(from_=i, to_=i+1, std=1., dim=2) for i in range(num_targets-1)],
#         reverses=[mk_mnlinear_kernel(from_=i+1, to_=i, std=1., dim=2) for i in range(num_targets-1)],

        # NOTES: With 1 intermediate density between 2 1d guassians with a linear kernel everything is fine
#         targets=[mk_n(i, i*2) for i in range(num_targets)],
#         forwards=[mk_nlinear_kernel(from_=i, to_=i+1, std=1., dim=1) for i in range(num_targets-1)],
#         reverses=[mk_nlinear_kernel(from_=i+1, to_=i, std=1., dim=1) for i in range(num_targets-1)],
    )


def sample_along(proposal, kernels, sample_shape=(2000,)):
    samples = []
    tr, _, out = proposal(sample_shape=sample_shape)
    samples.append(out)
    for k in kernels:
        proposal = Forward(k, proposal)
        tr, _, out = proposal(sample_shape=sample_shape)
        samples.append(out)
    return samples
