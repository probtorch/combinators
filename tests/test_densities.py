#!/usr/bin/env python
#
import torch
import torch.nn as nn
import logging
from torch import Tensor
from torch import distributions
from pytest import fixture, mark
from tqdm import trange

import combinators.trace.utils as trace_utils
from combinators import Forward, Reverse, Propose, Kernel
from combinators.nnets import LinearMap
from combinators.objectives import nvo_rkl
from combinators.densities import MultivariateNormal, Tempered
# from combinators.densities.kernels import MultivariateNormalLinearKernel

class MultivariateNormalKernel(Kernel):
    def __init__(
            self,
            ext_name:str,
            loc:Tensor,
            cov:Tensor,
            net:nn.Module,
        ):
        super().__init__()
        self.ext_name = ext_name
        self.dim_in = 2
        self.net = net

    def apply_kernel(self, trace, cond_trace, cond_output, sample_dims=None):
        sample_shape = cond_output.shape
        if sample_dims is not None and cond_output.shape[0] == 1 and len(cond_output.shape) == 2:
            cond_output = cond_output.T
        mu = self.net(cond_output.detach()).view(sample_shape)
        return trace.multivariate_normal(loc=mu,
                                         covariance_matrix=torch.eye(2),
                                         value=cond_trace[self.ext_name].value if self.ext_name in cond_trace else None,
                                         name=self.ext_name)

class MultivariateNormalLinearKernel(MultivariateNormalKernel):
    def __init__(self, ext_name:str, loc:Tensor, cov:Tensor):
        super().__init__(ext_name, loc, cov, LinearMap(dim=2))

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

@fixture(autouse=True)
def seed():
    torch.manual_seed(1)

def test_tempered_sampling(seed):
    num_targets = 3
    g0 = MultivariateNormal(name='g0', loc=torch.zeros(2), cov=torch.eye(2)**2)
    gK = MultivariateNormal(name=f'g{num_targets}', loc=torch.ones(2)*8, cov=torch.eye(2)**2)
    forwards = [MultivariateNormalLinearKernel(ext_name=f'g{i}', loc=torch.ones(2)*i, cov=torch.eye(2)) for i in range(1,num_targets+1)]
    reverses = [MultivariateNormalLinearKernel(ext_name=f'g{i}', loc=torch.ones(2)*i, cov=torch.eye(2)) for i in range(0,num_targets)]

    betas = torch.arange(0., 1., 1./num_targets)[1:] # g_0 is beta=0

    path = [Tempered(f'g{k}', g0, gK, beta) for k, beta in zip(range(1,num_targets), betas)]
    path = [g0] + path + [gK]
    targets = path

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = 10
    sample_shape = (7,)
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
                assert set(p_prv_tr.keys()) == { f'g{k+1}' }

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.target.trace[f'g{k+1}'])
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
