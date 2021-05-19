import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
import torch.distributions as D
from probtorch.stochastic import Trace, RandomVariable, ImproperRandomVariable, Provenance


class Resampler:
    def __call__(self, trace:Trace, log_weight:Tensor, sample_dims:int, batch_dim:int)->Tuple[Trace, Tensor]:
        raise NotImplementedError()

    def ancestor_indices_systematic(self, lw, sample_dims, batch_dim):
        raise NotImplementedError()

    def pick(self, z, aidx, sample_dims):
        ddim = z.dim() - aidx.dim()
        assert z.shape[:z.dim()-ddim] == aidx.shape, "data dims must be at the end of arg:z"

        mask = aidx
        for _ in range(ddim):
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(z)

        return z.gather(sample_dims, mask)


class Systematic(Resampler):
    def __init__(self, normalize_weights=False):
        super().__init__()
        self.normalize_weights = normalize_weights

    def __call__(self, trace:Trace, log_weight:Tensor, sample_dims:int, batch_dim:Optional[int])->Tuple[Trace, Tensor]:
        assert sample_dims == 0, "FIXME: take this assert out"

        _batch_dim = None
        if batch_dim is None:
            assert len(log_weight.shape) == 1, "batch_dim None requires 1d log_weight"
            _batch_dim = 1
            log_weight = log_weight.unsqueeze(_batch_dim)

        aidx = self.ancestor_indices_systematic(log_weight, sample_dims=sample_dims, batch_dim=_batch_dim if batch_dim is None else batch_dim)

        if batch_dim is None:
            aidx = aidx.squeeze(_batch_dim)

        new_trace = Trace()
        for key, rv in trace._nodes.items():
            # WARNING: Semantics only support resampling on traces (taus not rhos) which do not include OBSERVED RVs
            if not rv.resamplable or rv.provenance == Provenance.OBSERVED:
                new_trace._inject(rv, name=key, silent=True)
                continue

            # FIXME: Do not detach all
            value = self.pick(rv.value, aidx, sample_dims=sample_dims)
            log_prob = self.pick(rv.log_prob, aidx, sample_dims=sample_dims)

            if isinstance(rv, RandomVariable):
                var = RandomVariable(dist=rv._dist, value=value, log_prob=log_prob,
                                     provenance=rv.provenance, reparameterized=rv.reparameterized)
            elif isinstance(rv, ImproperRandomVariable):
                var = ImproperRandomVariable(log_density_fn=rv.log_density_fn, value=value, log_prob=log_prob,
                                             provenance=rv.provenance)
            else:
                raise NotImplementedError()

            new_trace._inject(var, name=key, silent=True)

        log_weight = torch.logsumexp(log_weight - math.log(log_weight.shape[sample_dims]), dim=sample_dims, keepdim=True).expand_as(log_weight)
        if self.normalize_weights:
            log_weight = torch.nn.functional.softmax(log_weight, dim=sample_dims).log()
        return new_trace, log_weight

    def ancestor_indices_systematic(self, lw, sample_dims, batch_dim):
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

