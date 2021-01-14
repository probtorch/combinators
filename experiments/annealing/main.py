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
from combinators.densities import MultivariateNormal, Tempered, RingGMM
from combinators.debug import print_grads
from combinators.densities.kernels import MultivariateNormalKernel, MultivariateNormalLinearKernel
from combinators.nnets import ResMLPJ
from combinators.objectives import nvo_rkl
from combinators import Forward, Reverse, Propose
from combinators.stochastic import RandomVariable, ImproperRandomVariable
from combinators.metrics import effective_sample_size, log_Z_hat
import visualize as V

# tensorutils
import pickle
import torch
from torch import Tensor
from typing import Optional, Union, Dict
from typeguard import typechecked
import base64
import hashlib
import os

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
    # return MultivariateNormalLinearKernel(
    #     ext_from=f'g{from_}',
    #     ext_to=f'g{to_}',
    #     loc=torch.zeros(2, **kw_autodevice()),
    #     cov=torch.eye(2, **kw_autodevice())*std**2)

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
        forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets-1)],
        reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets-1)],
    )

from combinators import Forward

def sample_along(proposal, kernels, sample_shape=(2000,)):
    samples = []
    tr, out = proposal(sample_shape=sample_shape)
    samples.append(out)
    for k in kernels:
        proposal = Forward(k, proposal)
        tr, out = proposal(sample_shape=sample_shape)
        samples.append(out)
    return samples

def _hash(t:Tensor, length:int):
    hasher = hashlib.sha1(pickle.dumps(t))
    return base64.urlsafe_b64encode(hasher.digest()[:length]).decode('ascii')

@typechecked
def thash(aten:Tensor, length:int=8, no_grad_char:str=" ")->str:
    g = "∇" if aten.requires_grad else no_grad_char
    save_ref = aten.detach()
    r = _hash(save_ref, length)
    # v = _hash(save_ref.clone(), 3*(length // 4))
    return f'#{g}{r}'

@typechecked
def show(aten:Tensor, fix_width:bool=True)->str:
    t = str(aten.dtype).split(".")[-1]
    s = "×".join(map(str, aten.shape))
    return f"{t}[{s}]{thash(aten)}"

def main(seed=1, num_iterations=1000, eval_break=500):
    # Setup
    torch.manual_seed(seed)
    K = 4
    num_samples = 256
    sample_shape=(num_samples,)

    # Models
    out = mk_model(K)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])
    f01, f12, f23 = forwards
    r10, r21, r32 = reverses

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    # with torch.no_grad():
    #     fp010, fp120, fp230 = fps0 = [[copy(p, requires_grad=False) for p in fwd.parameters()] for fwd in forwards]
    #     hashes_fps0 = [thash(f) for fs in fps0 for f in fs]


    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-3)
    optimizer.zero_grad()
    breakpoint();

    with trange(num_iterations) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros(1, **kw_autodevice())
            lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []
            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                # p_prv_tr = trace_utils.copytrace(p_prv_tr, requires_grad=RequiresGrad.NO)
                breakpoint();

                q.with_observations(p_prv_tr)
                q_ext = Forward(fwd, q, _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state, lv = extend(sample_shape=sample_shape, sample_dims=0)
                # breakpoint();

                p_prv_tr = state.target.trace

                p.clear_observations()
                q.clear_observations()

                lw += lv

                # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
                if k == 2:
                    loss = nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])

                lvss.append(lv)
            loss.backward()
            fp011, fp121, fp231 = fps1 = [[copy(p, requires_grad=False) for p in fwd.parameters()] for fwd in forwards]
            hashes_fps1 = [thash(f) for fs in fps1 for f in fs]
            print([[torch.equal(p0, p1) for p0, p1 in zip(p0s, p1s)] for p0s, p1s in zip(fps0, fps1)])
            print([p0 == p1 for p0, p1 in zip(hashes_fps0, hashes_fps1)])

            breakpoint();

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                # REPORTING
                # ---------------------------------------
                # loss
                loss_ct += 1
                loss_scalar = loss.detach().cpu().mean().item()
                writer.add_scalar('loss', loss_scalar, i)
                loss_sum += loss_scalar

                # ESS
                lvs = torch.stack(lvss, dim=0)
                lws = torch.cumsum(lvs, dim=1)
                ess = effective_sample_size(lws, sample_dims=-1)
                for step, x in zip(range(1,len(ess)+1), ess):
                    writer.add_scalar(f'ess/step-{step}', x, i)

                # logZhat
                lzh = log_Z_hat(lws, sample_dims=-1)
                for step, x in zip(range(1,len(lzh)+1), lzh):
                    writer.add_scalar(f'log_Z_hat/step-{step}', x, i)

                # progress bar
                if i % 10 == 0:
                    loss_avg = loss_sum / loss_ct
                    loss_template = 'loss={:3.4f}'.format(loss_avg)
                    lZh_template = "lZh:" + ";".join(['{:3.2f}'.format(lzh[i].cpu().item()) for i in range(len(lzh))])
                    ess_template = "ess:" + ";".join(['{:3.1f}'.format(ess[i].cpu().item()) for i in range(len(ess))])
                    loss_ct, loss_sum  = 0, 0.0
                    bar.set_postfix_str("; ".join([loss_template, ess_template, lZh_template]))

                # # show samples
                # if i % (eval_break-1) == 0:
                #     samples = sample_along(targets[0], forwards)
                #     fig = V.scatter_along(samples)
                #     writer.add_figure('overview', fig, global_step=i, close=True)
    with torch.no_grad():
        xss = sample_along(targets[0], forwards)
        print([xs.mean() for xs in xss])
        breakpoint();
        print()




if __name__ == '__main__':
    main(num_iterations=100)
