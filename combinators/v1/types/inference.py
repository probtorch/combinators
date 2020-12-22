"""
Old Inference Combinators

z_1, \tau_1, w_1 <~~ g(x_0)            z_2, \tau_2, w_2 <~~ f(x_0 ; \tau_1)
--------------------------------------------------------------------------- (Inf-Propose)
z_2, \tau_2, frac{w_1 w_2}{w_g (\tau_1;\tau_2, x_0)} <~~ propose(f, g)(x_0)


z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
-------------------------------------------------------------------------------------------- (Inf-Resample)
z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)


(Inf-Move):
z_1, \tau_1, w_1  <~~  f(x_0)            z_2, \tau_2, w_2  <~~  g(z_1)             z_3, \tau_3, w_3  <~~  f(x_0; \tau_2)
----------------------------------------------------------------------------------------------------------------------
                  z_3, \tau_3, w_1 \frac{w_2 w_3}{w_g(\tau_2;\tau_3, z_1)} <~~ move(f,g)(x_0)

"""
import torch
from probtorch import Trace
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typeguard import typechecked
from typing import Tuple, NewType, Any, List, Callable

from combinators.types.combinator import Combinator, Output, Weight
from combinators.types.model import Program
from combinators.types.kernel import Kernel


class Inference(Combinator, metaclass=ABCMeta):
    """
    Superclass for
    f := "unnormalized generator :: i -> ctx_θ γ" | "normalized generator :: i -> ctx_θ π"

    k := "kernel program :: f -> ctx ( xs ~ π )"                       -- NOTE: in NVI: π = distribution
                                 ^^^
                                    `--- log probs, samples, (validating traces?),

    p := reverse(p, k) | f | k
    q := propose(p, q) | resample(q) | forward(k, q) | p
    """

    @abstractmethod
    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...


class TargetExpr(Inference):
    """ p := reverse(p, k) | f | k """
    ...


class Reverse(TargetExpr):
    """ p := reverse(p, k) | ... """

    def __init__(self, target: TargetExpr, kernel: Kernel):
        self.target = target
        self.kernel = kernel

    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...

    def __repr__(self) -> str:
        return "Reverse({}, {})".format(repr(self.target), repr(self.kernel))


class Target_P(TargetExpr):
    """ p := ... | f | ... """

    def __init__(self, f: Program):
        self.f = f

    def __call__(self, *args, **varargs) -> Tuple[Output, Trace, Weight]:
        z = self.f(*args, **varargs)
        trace = None
        weight = torch.ones(z.shape)
        return z, trace, weight

    def __repr__(self) -> str:
        return "Target_P({})".format(repr(self.f))


class Kernel_P(TargetExpr):
    """ p := ... | k """

    def __init__(self, k: Kernel):
        self.k = k

    def __call__(self, prgm: Program, *args, **varargs) -> Tuple[Output, Trace, Weight]:
        z = self.k(prgm, *args, **varargs)
        trace = None
        weight = None
        return z, trace, weight

    def __repr__(self) -> str:
        return "Kernel_P({})".format(repr(self.k))


class ProposalExpr(Inference):
    """ q := propose(p, q) | resample(q) | forward(k, q) | p """
    ...


class Propose(ProposalExpr):
    """ q := propose(p, q) | ...  """

    def __init__(self, p: TargetExpr, q: ProposalExpr):
        self.p, self.q = p,
        self.q = q

    def __call__(self, target: TargetExpr, proposal: ProposalExpr) -> Tuple[Output, Trace, Weight]:
        ...

    def __repr__(self) -> str:
        return "Propose({})".format(repr(self.k))


class Resample(ProposalExpr):
    """ q := ... | resample(q) | ...  """

    def __init__(self, proposal: ProposalExpr):
        self.proposal = proposal

    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...


class Forward(ProposalExpr):
    """ q := ... | forward(k, q) | ...  """

    def __init__(self, kernel: KernelProtocol, proposal: ProposalExpr):
        self.kernel = kernel
        self.proposal = proposal

    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...


class Target_Q(ProposalExpr):
    """ q := ... | p """

    def __init__(self, target: TargetExpr):
        self.target = target

    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        return self.target(samples)
