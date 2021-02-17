import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Tuple
import math
import torch.distributions as D
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from combinators.tensor.utils import kw_autodevice
from combinators.stochastic import Trace, RandomVariable, ImproperRandomVariable, Provenance
import combinators.stochastic as probtorch

class Strategy:
    def __call__(self, trace:Trace, log_weight:Tensor, sample_dims:int, batch_dim:int)->Tuple[Trace, Tensor]:
        raise NotImplementedError()

def pick(z, aidx, sample_dims):
    ddim = z.dim() - aidx.dim()
    assert z.shape[:z.dim()-ddim] == aidx.shape, "data dims must be at the end of arg:z"

    mask = aidx
    for _ in range(ddim):
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(z)

    return z.gather(sample_dims, mask)

def ancestor_indices_systematic(lw, sample_dims, batch_dim):
    assert batch_dim is not None and sample_dims is not None
    _sample_dims = -1
    n, b = lw.shape[sample_dims], lw.shape[batch_dim]

    u = torch.rand(b, device=lw.device)
    usteps = torch.stack([(k + u) for k in range(n)], dim=_sample_dims)/n
    nws = F.softmax(lw.detach(), dim=sample_dims)

    csum = nws.transpose(sample_dims, _sample_dims).cumsum(dim=_sample_dims)
    cmax, _ = torch.max(csum, dim=_sample_dims, keepdim=True)
    ncsum = csum / cmax

    aidx = torch.searchsorted(ncsum, usteps, right=False)
    aidx = aidx.transpose(_sample_dims, sample_dims)
    return aidx

class Systematic(Strategy):
    def __init__(self, quiet=False, normalize_weights=False):
        self.quiet=quiet
        self.normalize_weights = normalize_weights

    def __call__(self, trace:Trace, log_weight:Tensor, sample_dims:int, batch_dim:Optional[int])->Tuple[Trace, Tensor]:
        assert sample_dims == 0, "FIXME: take this assert out"

        if batch_dim is None:
            assert len(log_weight.shape) == 1, "batch_dim None requires 1d log_weight"
            _batch_dim = 1
            log_weight = log_weight.unsqueeze(_batch_dim)

        aidx = ancestor_indices_systematic(log_weight, sample_dims=sample_dims, batch_dim=_batch_dim if batch_dim is None else batch_dim)

        if batch_dim is None:
            aidx = aidx.squeeze(_batch_dim)

        new_trace = Trace()
        for key, rv in trace._nodes.items():
            # WARNING: Semantics only support resampling on traces (taus not rhos) which do not include OBSERVED RVs
            if not rv.can_resample or rv.provenance == Provenance.OBSERVED:
                new_trace.append(rv, name=key)
                continue

            # FIXME: Do not detach all
            value = pick(rv.value, aidx, sample_dims=sample_dims)
            log_prob = pick(rv.log_prob, aidx, sample_dims=sample_dims)

            if isinstance(rv, RandomVariable):
                var = RandomVariable(dist=rv._dist, value=value, log_prob=log_prob,
                                     provenance=rv.provenance, reparameterized=rv.reparameterized)
            elif isinstance(rv, ImproperRandomVariable):
                var = ImproperRandomVariable(log_density_fn=rv.log_density_fn, value=value, log_prob=log_prob,
                                             provenance=rv.provenance)
            else:
                raise NotImplementedError()

            new_trace.append(var, name=key)

        log_weight = torch.logsumexp(log_weight - math.log(log_weight.shape[sample_dims]), dim=sample_dims, keepdim=True).expand_as(log_weight)
        if self.normalize_weights:
            log_weight = torch.nn.functional.softmax(log_weight, dim=sample_dims).log()
        return new_trace, log_weight

def stubtest_ancestor_indices_systematic():
    S = 4
    B = 1000
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
    lw = lw.unsqueeze(1).expand(S, B)
    a = ancestor_indices_systematic(lw, 0, 1).T
    for i in range(S):
        print(i, ( a == i ).sum() / (S*B))

def stubtest_resample_with_batch(B=100, N=5):
    S = 4

    value = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()

    lw = lw.unsqueeze(1).expand(S, B)
    value = value.unsqueeze(1).expand(S, B, 2)
    tr = Trace()

    for n in range(N):
        tr.append(RandomVariable(dist=D.Normal(0, 1), value=value, log_prob=lw), name=f'z_{n}')

    resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=1)
    assert (_lw.exp() == 0.25).all()

    memo = torch.zeros(S)
    for n, (_, rv) in enumerate(resampled.items()):
        for s in range(S):
            memo[s] += (rv.value == (s+1)).sum() / (S*B*N*2)

    print(memo)

def stubtest_resample_without_batch():
    S = 4
    N = 5
    B = 100

    value = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
    tr = Trace()

    memo = torch.zeros(S)
    for _ in range(B):
        for n in range(N):
            tr.append(RandomVariable(dist=D.Normal(0, 1), value=value, log_prob=lw), name=f'z_{n}')

        resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=None)

        assert (_lw.exp() == 0.25).all()
        for n, (_, rv) in enumerate(resampled.items()):
            for s in range(S):
                memo[s] += (rv.value == (s+1)).sum() / (S*N*2)

    print(memo / B)

