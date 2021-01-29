#!/usr/bin/env python3

import torch
from torch import Tensor
from torch import nn
import torch.distributions as dist
from combinators.inference import State
from combinators.stochastic import Trace
from typeguard import typechecked
from typing import Callable
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

class MaybeWriter(SummaryWriter):
    def __init__(self, enable=True):
        super().__init__()
        self.enabled = enable

    def maybe_run(self, function, *args, **kwargs):
        if self.enabled:
            return function(self, *args, **kwargs)

    def add_scalar(self,*args, **kwargs):
        self.maybe_run(super().add_scalar, *args, **kwargs)

    def add_figure(self,*args, **kwargs):
        self.maybe_run(super().add_figure, *args, **kwargs)

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

def seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = True # just incase something goes wrong with set_deterministic

def print_grad(*args:nn.Module):
    for i, x in enumerate(*args):
        for j, param in enumerate(x.parameters()):
            print(i, j, param.grad)

def excise_state(state:State) -> State:
    tr = excise_trace(state.trace)
    o = excise(state.output) if isinstance(state.output, Tensor) else state.output
    return State(tr, o)

def print_grads(learnables, bools_only=True):
    for i, k in enumerate(learnables):
        for j, p in enumerate(k.parameters()):
            print(i, j, "none" if p is None or torch.all(p == 0) else ('exists' if bools_only else p))

def propagate(N:dist.MultivariateNormal, F:Tensor, t:Tensor, B:Tensor, marginalize:bool=False, reverse_order:bool=False)-> dist.MultivariateNormal:
    # N is normal starting from
    F = F.cpu() # F is NN weights on linear network of forward kernel
    t = t.cpu() # t is bias
    B = B.cpu() # b is cov of kernel
    with torch.no_grad():
        a = N.loc.cpu()
        A = N.covariance_matrix.cpu()
        b = t + F @ a
        m = torch.cat((a, b))
        FA = F @ A
        BFFA = B + F @ (FA).T
        if marginalize:
            return dist.MultivariateNormal(loc=b, covariance_matrix=BFFA)
        if not reverse_order:
            A = N.covariance_matrix.cpu()
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
