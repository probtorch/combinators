
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike
from combinators.utils import assert_valid_subtrace, copytrace



@typechecked
class Traceable(ABC):
    """ superclass of a program? """
    def __init__(self) -> None:
        super().__init__()
        self._trace: Optional[Trace] = None
        self.observations:Dict[str, Tensor] = dict()

    def log_probs(self, values:Optional[TraceLike] = None) -> Dict[str, Tensor]:
        def eval_under(getter, k:str) -> Tensor:
            is_given = values is not None and k in values.keys()
            log_prob = self._trace[k].dist.log_prob(getter(values, k)) if is_given else self._trace[k].log_prob
            return log_prob

        if values is None:
            return {k: self._trace[k].log_prob for k in self.variables}

        elif isinstance(values, dict):
            return {k: eval_under(lambda vals, k: vals[k], k) for k in self.variables}

        elif isinstance(values, Trace):
            return {k: eval_under(lambda vals, k: vals[k].value, k) for k in self.variables}

    def get_trace(self, evict=False) -> Trace:
        # FIXME: traces are super complicated now : (
        self._trace = Trace() if self._trace is None or evict else self._trace

        return self._trace

    def sample(self, dist:Any, value:Tensor, name:Optional[str]=None) -> None:
        raise NotImplementedError("requires different event handling structure.")

    def observe(self, key:str, value:Tensor) -> None:
        self.observations[key] = value

    def _apply_observes(self, trace:Trace) -> Trace:
        for key, value in self.observations.items():
            trace.enqueue_observation(key, value)
        return trace

    @property
    def variables(self) -> Set[str]:
        return set(self.get_trace().keys())

