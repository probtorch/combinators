#!/usr/bin/env python3
"""
Old Inference Combinators

z_1, \tau_1, w_1 <~~ g(x_0)            z_2, \tau_2, w_2 <~~ f(x_0 ; \tau_1)
--------------------------------------------------------------------------- (Inf-Propose)
z_2, \tau_2, frac{w_1 w_2}{w_g (\tau_1;\tau_2, x_0)} <~~ propose(f, g)(x_0)


z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
-------------------------------------------------------------------------------------------- (Inf-Resample)
z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)


z_1, \tau_1, w_1  <~~  f(x_0)            z_2, \tau_2, w_2  <~~  g(z_1)             z_3, \tau_3, w_3  <~~  f(x_0; \tau_2)
----------------------------------------------------------------------------------------------------------------------- (Inf-Move)
                  z_3, \tau_3, w_1 \frac{w_2 w_3}{w_g(\tau_2;\tau_3, z_1)} <~~ move(f,g)(x_0)

"""
import torch
from probtorch import Trace
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typeguard import typechecked
from typing import Tuple, NewType, Any, List, Callable, Protocol

Output = NewType("Output", Any)
Weight = NewType("Weight", Tensor)
Program = NewType("Program", Callable[..., Output])
Kernel = NewType("Kernel", Callable[..., Output])


class KernelProtocol(Protocol):
    """
    Python type-hints don't support things like "the first argument is X followed by varargs" -- the closest thing
    is a Protocol. Kernels take in a program and varargs (which we include for pragmatics) and return output of a tensor
    """

    def __call__(self, prgm: Program, **kwargs) -> Output:
        ...


def kernel_protocol(func: KernelProtocol):
    """ decorator for typechecked kernel protocols """
    ...


class Combinator(metaclass=ABCMeta):
    """
    A superclass for inference combinators (for maybe involving model combinators)
    """
    @abstractmethod
    @typechecked
    def __call__(self, *args: Any) -> Tuple[Output, Trace, Weight]:
        ...


class Inference(Combinator, metaclass=ABCMeta):
    """
    Superclass for
    k := "kernel program"
    f := "target program"
    p := reverse(p, k) | f | k
    q := propose(p, q) | resample(q) | forward(k, q) | p
    """

    def __call__(self, samples: Tensor) -> Tuple[Output, Trace, Weight]:
        ...


class TargetExpr(Inference):
    """ p := reverse(p, k) | f | k """
    ...


class Reverse(TargetExpr):
    """ p := reverse(p, k) | ... """

    def __call__(self, target: TargetExpr, kernel: KernelProtocol) -> Tuple[Output, Trace, Weight]:
        ...


class Target(TargetExpr):
    """ p := ... | f | ... """

    def __init__(self, f: Program):
        self.f = f

    def __call__(self, *args, **varargs) -> Tuple[Output, Trace, Weight]:
        z = self.f(*args, **varargs)
        trace = None
        weight = torch.ones(z.shape)
        return z, trace, weight


class Kernel(TargetExpr):
    """ p := ... | k """

    def __init__(self, k: Kernel):
        self.k = k

    def __call__(self, prgm: Program, *args, **varargs) -> Tuple[Output, Trace, Weight]:
        z = self.k(prgm, *args, **varargs)
        trace = None
        weight = None
        return z, trace, weight


class ProposalExpr(Inference):
    """ q := propose(p, q) | resample(q) | forward(k, q) | p """
    ...


class Propose(ProposalExpr):
    """ q := propose(p, q) | ...  """

    def __call__(self, target: TargetExpr, proposal: ProposalExpr) -> Tuple[Output, Trace, Weight]:
        ...


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


# ======================================


class Reweight(Inference):
    """
    z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
    -------------------------------------------------------------------------------------------- (Inf-Resample)
    z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)
    """

    def __init__(self, f, K):
        self.f, self.K = f, K

    @typechecked
    def __call__(self, alpha, x) -> Tuple[Tensor, Trace, Weight]:
        zs, traces, weights = self.f(alpha, x)
        discrete_samples = self.Discrete(weights)
        return zs, traces, (1 / self.K) * weights.sum()  # ???


def reweight(
        f: Callable[[Tensor, Tensor], Tuple[List[Tensor], List[Trace], List[Weight]]],
        K: int,
) -> Tuple[Tensor, Trace, Weight]:

    return Reweight(f, K)


class Propose(Inference):
    """
      z_1, \tau_1, w_1 <~~ g(x_0)            z_2, \tau_2, w_2 <~~ f(x_0 ; \tau_1)
      --------------------------------------------------------------------------- (Inf-Propose)
      z_2, \tau_2, frac{w_1 w_2}{w_g (\tau_1;\tau_2, x_0)} <~~ propose(f, g)(x_0)

      z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
      -------------------------------------------------------------------------------------------- (Inf-Resample)
      z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)

      z_1, \tau_1, w_1  <~~  f(x_0)            z_2, \tau_2, w_2  <~~  g(z_1)             z_3, \tau_3, w_3  <~~  f(x_0; \tau_2)
      ----------------------------------------------------------------------------------------------------------------------- (Inf-Move)
                        z_3, \tau_3, w_1 \frac{w_2 w_3}{w_g(\tau_2;\tau_3, z_1)} <~~ move(f,g)(x_0)
    """
    def __init__(self):
        pass


def propose():
    pass


class Extend(Inference):
    """
    z_1, \tau_1, w_1 <~~ g(x_0)            z_2, \tau_2, w_2 <~~ f(x_0 ; \tau_1)
    --------------------------------------------------------------------------- (Inf-Propose)
    z_2, \tau_2, frac{w_1 w_2}{w_g (\tau_1;\tau_2, x_0)} <~~ propose(f, g)(x_0)

    z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
    -------------------------------------------------------------------------------------------- (Inf-Resample)
    z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)

    z_1, \tau_1, w_1  <~~  f(x_0)            z_2, \tau_2, w_2  <~~  g(z_1)             z_3, \tau_3, w_3  <~~  f(x_0; \tau_2)
    ----------------------------------------------------------------------------------------------------------------------- (Inf-Move)
                      z_3, \tau_3, w_1 \frac{w_2 w_3}{w_g(\tau_2;\tau_3, z_1)} <~~ move(f,g)(x_0)
    """
    def __init__(self):
        pass


def extend():
    pass


class Move(Inference):
    """
      z_1, \tau_1, w_1 <~~ g(x_0)            z_2, \tau_2, w_2 <~~ f(x_0 ; \tau_1)
      --------------------------------------------------------------------------- (Inf-Propose)
      z_2, \tau_2, frac{w_1 w_2}{w_g (\tau_1;\tau_2, x_0)} <~~ propose(f, g)(x_0)

      z^k, \tau^k, w^k <~~ f(\alpha, x)      a^1 ... a^K ~ Discrete(w^1...w^K)      | for(k=1...K)
      -------------------------------------------------------------------------------------------- (Inf-Resample)
      z^{a^k}, \tau^{a^k}, (1/K) \\sum_k w^k <~~ resample(f, K)(\alpha, x)           | for(k=1...K)

      z_1, \tau_1, w_1  <~~  f(x_0)            z_2, \tau_2, w_2  <~~  g(z_1)             z_3, \tau_3, w_3  <~~  f(x_0; \tau_2)
      ----------------------------------------------------------------------------------------------------------------------- (Inf-Move)
                        z_3, \tau_3, w_1 \frac{w_2 w_3}{w_g(\tau_2;\tau_3, z_1)} <~~ move(f,g)(x_0)
    """
    def __init__(self):
        pass


def move():
    pass


if __name__ == '__main__':
    print('success!')
