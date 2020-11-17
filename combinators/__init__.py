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
from probtorch import Trace
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typeguard import typechecked
from typing import Tuple, NewType, Any, List, Callable

Weight = NewType("Weight", Tensor)


class Combinator(metaclass=ABCMeta):

    @abstractmethod
    @typechecked
    def __call__(self, left: Any, right: Any) -> Tuple[Tensor, Trace, Weight]:
        ...


class Model(Combinator):
    pass


class Inference(Combinator):
    pass


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
