from enum import Enum, auto
import math
import torch

class CovarianceEmbedding(Enum):
    LogDiagonal = auto()
    LowerCholesky = auto()
    SoftPlusDiagonal = auto()

    def embed(self, cov, dim):
        """
        asd
        """
        assert cov.shape == (dim, dim)
        if self == CovarianceEmbedding.LogDiagonal:
            return cov.diag().log()

        elif self == CovarianceEmbedding.SoftPlusDiagonal:
            return cov.diag().expm1().log()

        elif self == CovarianceEmbedding.LowerCholesky:
            tidx = torch.tril_indices(row=dim, col=dim, offset=0)
            return torch.cholesky(cov, upper=False)[tidx[0], tidx[1]]

        raise RuntimeError()

    def unembed(self, emb, dim):
        if self == CovarianceEmbedding.LogDiagonal:
            assert emb.shape[-1:] == (dim,)  # ignore batch shape
            return torch.diag_embed(emb.exp())

        elif self == CovarianceEmbedding.SoftPlusDiagonal:
            assert emb.shape[-1:] == (dim,)  # ignore batch shape
            return torch.diag_embed(torch.nn.functional.softplus(emb))

        elif self == CovarianceEmbedding.LowerCholesky:
            assert dim == int((math.sqrt(8*emb.shape[-1]+1) - 1)/2)
            tidx = torch.tril_indices(row=dim, col=dim, offset=0)
            if torch.cuda.is_available():
                L = torch.cuda.FloatTensor((*emb.shape[:-1], dim, dim)).fill_(0)
            else:
                L = torch.FloatTensor((*emb.shape[:-1], dim, dim)).fill_(0)
            L[..., tidx[0], tidx[1]] = emb
            return torch.matmul(L, L.transpose(-1, -2))
        raise RuntimeError()

    @property
    def embed_name(self):
        if self == CovarianceEmbedding.LogDiagonal:
            return 'cov_log_diagonal'
        elif self == CovarianceEmbedding.SoftPlusDiagonal:
            return 'cov_softplus_diagonal'
        elif self == CovarianceEmbedding.LowerCholesky:
            return 'cov_lower_cholesky'

    def embed_dim(self, dim):
        if self == CovarianceEmbedding.LogDiagonal or self == CovarianceEmbedding.SoftPlusDiagonal:
            return dim
        elif self == CovarianceEmbedding.LowerCholesky:
            return (dim**2 + dim)/2
