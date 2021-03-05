#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import Tensor


def effective_sample_size(lw: Tensor, sample_dims=-1) -> Tensor:
    lnw = F.softmax(lw, dim=sample_dims).log()
    ess = torch.exp(-torch.logsumexp(2*lnw, dim=sample_dims))
    return ess


def log_Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return Z_hat(lw, sample_dims=sample_dims).log()


def Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return lw.exp().mean(dim=sample_dims)
