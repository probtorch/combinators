import torch
from torch.distributions import Categorical

def resample_systematic(z, lw, sample_dim=0, batch_dim=1):
    aidx = ancestor_indices_systematic(lw, sample_dim=sample_dim, batch_dim=batch_dim)
    z = pick(z, aidx)
    lw = torch.logsumexp(lw - lw[sample_dim], dim=sample_dim).expand_as(lw)
    return z, lw

def pick(z, idx, sample_dim=0):
    ddim = z.dim() - idx.dim()
    mask = idx[(...,) + (None,)*ddim].expand_as(z)
    return z.gather(sample_dim, mask)

def ancestor_indices_systematic(lw, sample_dim=0, batch_dim=1, strategy='systematic'):
    if batch_dim is None:
        _batch_dim = 1
        lw = lw[None, :]  # Add empty batch dim
    else:
        _batch_dim = batch_dim
    n, b = lw.shape[sample_dim], lw.shape[_batch_dim]
    u = torch.rand(b)
    usteps = torch.stack([(k + u) for k in range(n)], dim=_batch_dim)/n
    nws = torch.nn.functional.softmax(lw.detach(), dim=sample_dim)
    csum = nws.transpose(sample_dim, -1).cumsum(dim=-1)
    cmax, _ = torch.max(csum, dim=_batch_dim, keepdim=True)
    ncsum = csum / cmax
    aidx = torch.searchsorted(ncsum, usteps, right=False)
    aidx = aidx.transpose(-1, sample_dim)
    if batch_dim is None:
        aidx = aidx.squeeze()  # remove empty batch dim
    return aidx

def ancestor_indices_multinomial(lw, sample_dim=0, batch_dim=1, strategy='systematic'):
    # n, b = lw.shape[sample_dim], lw.shape[batch_dim]
    nws = torch.nn.functional.softmax(lw, dim=sample_dim)
    aidx = Categorical(nws.transpose(sample_dim, -1)).sample((1, ))
    return aidx
