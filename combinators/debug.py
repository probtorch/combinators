#!/usr/bin/env python3

import os
import torch
from torch import Tensor
import torch.distributions as dist
from typeguard import typechecked
import numpy as np
import random


@typechecked
def runtime() -> str:
    try:
        # magic global function in ipython shells
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
        else:
            raise Exception()
    except:
        return 'terminal'

@typechecked
def seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = True # just incase something goes wrong with set_deterministic


def is_smoketest()->bool:
    env_var = os.getenv('SMOKE')
    return env_var is not None and env_var == 'true'


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

