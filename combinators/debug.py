#!/usr/bin/env python3

import torch
from torch import Tensor
import torch.distributions as dist
from combinators.inference import State
from combinators.stochastic import Trace
from typeguard import typechecked
from typing import Callable

def excise(t:Tensor) -> Tensor:
    """ clone a tensor and remove it from the computation graph """
    return torch.clone(t).detach()

def excise_trace(tr:Trace) -> Trace:
    """ deep-copy a trace and remove all tensors from the computation graph. """
    newtr = Trace()
    for k, rv in tr.items():
        RVClass = type(rv)
        newval = excise(rv.value)
        # FIXME: should also excise rv.dist parameters, but this isn't necessary for testing yet
        newrv = RVClass(rv.dist, newval, provenance=rv.provenance, mask=rv.mask)
        newtr.append(newrv, name=k)
    return newtr

def excise_state(state:State) -> State:
    tr = excise_trace(state.trace)
    o = excise(state.output) if isinstance(state.output, Tensor) else state.output
    return State(tr, o)


def propagate(N:dist.MultivariateNormal, F:Tensor, t:Tensor, B:Tensor, marginalize:bool=False, reverse_order:bool=False)-> dist.MultivariateNormal:
    # N is normal starting from
    # F is NN weights on linear network of forward kernel
    # t is bias
    # b is cov of kernel
    with torch.no_grad():
        a = N.loc
        A = N.covariance_matrix
        b = t + F @ a
        m = torch.cat((a, b))
        FA = F @ A
        BFFA = B + F @ (FA).T
        if marginalize:
            return dist.MultivariateNormal(loc=b, covariance_matrix=BFFA)
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
        return dist.MultivariateNormal(loc=m, covariance_matrix=C)

def empirical_marginal_mean_std(runnable:Callable[[], Tensor], num_validate_samples = 400):
    with torch.no_grad():
        samples = []
        for _ in range(num_validate_samples):
            out = runnable()
            samples.append(out)
        evaluation = torch.cat(samples)
        return evaluation.mean().item(), evaluation.std().item()
