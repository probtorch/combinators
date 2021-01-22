import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from combinators.stochastic import Trace, RandomVariable, ImproperRandomVariable

class Strategy:
    def __call__(self, trace:Trace, log_weight:Tensor)->Tuple[Trace, Tensor]:
        raise NotImplementedError()

    def _(self, trace:Trace, log_weight:Tensor)->None:
        raise NotImplementedError()

def pick(z, idx, sample_dim=0):
    ddim = z.dim() - idx.dim()
    mask = idx[(...,) + (None,)*ddim].expand_as(z)
    return z.gather(sample_dim, mask)

def ancestor_indices_systematic(lw, sample_dim=0, batch_dim=1):
    if batch_dim is None:
        _batch_dim = 1
        lw = lw[None, :]  # Add empty batch dim
    else:
        _batch_dim = batch_dim
    n, b = lw.shape[sample_dim], lw.shape[_batch_dim]
    u = torch.rand(b)
    usteps = torch.stack([(k + u) for k in range(n)], dim=_batch_dim)/n
    nws = F.softmax(lw.detach(), dim=sample_dim)
    csum = nws.transpose(sample_dim, -1).cumsum(dim=-1)
    cmax, _ = torch.max(csum, dim=_batch_dim, keepdim=True)
    ncsum = csum / cmax
    aidx = torch.searchsorted(ncsum, usteps, right=False)
    aidx = aidx.transpose(-1, sample_dim)
    if batch_dim is None:
        aidx = aidx.squeeze()  # remove empty batch dim
    return aidx

class Systematic(Strategy):
    def __init__(self, sample_dim=0, batch_dim=1):
        # self.sample_dim = sample_dim
        # self.batch_dim = batch_dim
        pass

    def __call__(self, trace, log_weight, sample_dim=0, batch_dim=1)->Tuple[Trace, Tensor]:
        # do we want to reason about these arguments at a higher level?
        self.sample_dim = sample_dim
        self.batch_dim = batch_dim

        aidx = ancestor_indices_systematic(log_weight, sample_dim=self.sample_dim, batch_dim=self.batch_dim)
        new_trace = Trace()
        for key, rv in trace._nodes.items():
            # TODO: Do not detach all
            value = pick(rv.value, aidx, sample_dim=self.sample_dim).detach()
            log_prob = pick(rv.log_prob, aidx, sample_dim=self.sample_dim).detach()

            if isinstance(rv, RandomVariable):
                var = RandomVariable(dist=rv._dist, value=value, log_prob=log_prob)
            elif isinstance(rv, ImproperRandomVariable):
                var = ImproperRandomVariable(log_density_fn=rv.log_density_fn, value=value, log_prob=log_prob)
            else:
                raise NotImplementedError()

            new_trace.append(var, name=key)
        log_weight = torch.logsumexp(log_weight - log_weight.shape[self.sample_dim], dim=self.sample_dim, keepdim=True).expand_as(log_weight)
        return new_trace, log_weight

    def _(self, trace, log_weight):
        """ inplace version """
        aidx = ancestor_indices_systematic(log_weight, sample_dim=self.sample_dim, batch_dim=self.batch_dim)
        for key in trace._nodes:
            trace[key]._value = pick(trace[key].value, aidx, sample_dim=self.sample_dim)
            trace[key]._log_prob = pick(trace[key].log_prob, aidx, sample_dim=self.sample_dim)
        log_weight = torch.logsumexp(log_weight, dim=self.sample_dim, keepdim=True).expand_as(log_weight)
        return trace, log_weight
