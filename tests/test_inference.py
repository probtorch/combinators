#!/usr/bin/env python
import torch
import torch.nn as nn
import logging
from torch import Tensor
from probtorch.util import expand_inputs
from collections import namedtuple
from typeguard import typechecked
from tqdm import trange
from pytest import mark, fixture
from typing import Optional

import combinators.trace.utils as trace_utils
from combinators.densities import Normal, MultivariateNormal
from combinators.nnets import LinearMap
from combinators.tensor.utils import thash, show
from combinators.inference import PCache # temporary
from combinators.stochastic import RandomVariable, Provenance
from combinators import Program, Kernel, Trace, Forward, Reverse, Propose

logger = logging.getLogger(__name__)

Tolerance = namedtuple("Tolerance", ['mean', 'std'])
Params = namedtuple("Params", ["mean", "std"])

def propagate(N, F, t, B, marginalize=False, reverse_order=False):
    # N is normal starting from
    # F is NN weights on linear network of forward kernel
    # t is bias
    # b is cov of kernel
    a = N.loc
    A = N.covariance_matrix
    b = t + F @ a
    m = torch.cat((a, b))
    FA = F @ A
    BFFA = B + F @ (FA).T
    if marginalize:
        return MultivariateNormal(loc=b, covariance_matrix=BFFA)
    if not reverse_order:
        A = N.covariance_matrix
        C1 = torch.cat((A, (FA).T), dim=1)
        C2 = torch.cat((FA, BFFA), dim=1)
        C = torch.cat((C1, C2), dim=0)
    if reverse_order:
        C1 = torch.cat((BFFA, FA), dim=1)
        C2 = torch.cat(((FA).T, A), dim=1)
        C = torch.cat((C1, C2), dim=0)
        m = torch.cat((b, a))
    return MultivariateNormal(loc=m, covariance_matrix=C)

def eval_mean_std(runnable, target_params:Params, tolerances:Tolerance, num_validate_samples = 400):
    with torch.no_grad():
        samples = []
        for _ in range(num_validate_samples):
            out = runnable()
            samples.append(out)
        evaluation = torch.cat(samples)
        eval_mean, eval_stdv = evaluation.mean().item(), evaluation.std().item()
        print("mean: {:.4f}, std: {:.4f}".format(eval_mean, eval_stdv))
        assert (target_params.mean - tolerances.mean) < eval_mean and  eval_mean < (target_params.mean + tolerances.mean)
        assert (target_params.std  - tolerances.std ) < eval_stdv and  eval_stdv < (target_params.std  + tolerances.std )

class LinearKernel(Kernel):
    def __init__(self, ext_name):
        super().__init__()
        self.net = LinearMap(dim=1)
        self.ext_name = ext_name

    def apply_kernel(self, trace, cond_trace, cond_output):
        mu = self.net(cond_output.detach())
        return trace.normal(loc=mu, scale=torch.ones_like(mu), name=self.ext_name)

    def __repr__(self):
        return f'ext={self.ext_name}:' + super().__repr__()

@fixture(autouse=True)
def scaffolding():
    torch.manual_seed(1)

# @mark.skip("this is not a problem anymore because of the observation clearing -- not sure if this is a problem yet...")
def test_independent_compute_graphs():
    g = lambda i: f"g{i}"
    g1, g2, g3 = targets  = [Normal(loc=i, scale=1, name=g(i)) for i in range(1,4) ]
    f12, f23   = forwards = [LinearKernel(ext_name=g(i)) for i in range(2,4) ]
    r21, r32   = reverses = [LinearKernel(ext_name=g(i)) for i in range(1,3) ]

    lr=1e-2
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*targets, *forwards, *reverses]], lr=lr)

    q12 = Forward(f12, g1)
    p21 = Reverse(g2, r21)
    extend12 = Propose(target=p21, proposal=q12)
    state12, lv12 = extend12(sample_shape=(50,1))
    import ipdb; ipdb.set_trace();


    for k, prg in [(g(1), state12.target), (g(2), state12.target), (g(2), state12.proposal)]:
        assert k == k and prg is prg and prg.trace[k].value.requiresgrad # k==k for debugging the assert

    assert not state12.proposal.trace[g(1)].value.requiresgrad
    analytic_forward_marginal2 = propagate(g1, f12.net.weight, f12.net.bias, torch.eye(1), marginalize=True)
    import ipdb; ipdb.set_trace();

    q23 = Forward(f23, g2)
    p32 = Reverse(g3, r32)
    extend23 = Propose(target=p32, proposal=q23)
    state23, lv23 = extend23()

    for k, prg in [(g(1), state23.target), (g(2), state23.target), (g(2), state23.proposal)]:
        assert k == k and prg is prg and prg.trace[k].value.requiresgrad # k==k for debugging the assert

    assert not state23.proposal.trace[g(1)].value.requiresgrad
    analytic_forward_marginal3 = propagate(g2, f23.net.weight, f23.net.bias, torch.eye(1), marginalize=True)

    nvo_avo(lv23)
