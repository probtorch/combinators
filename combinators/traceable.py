import logging
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
from copy import deepcopy

from combinators.stochastic import Trace, Factor
from combinators.types import Output, State, TraceLike, get_value
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils

logger = logging.getLogger(__name__)

@typechecked
class Conditionable(ABC):
    def __init__(self) -> None:
        super().__init__()
        # Optional is used to tell us if a program has a conditioning trace, as opposed to
        # a misspecified condition on the empty trace
        self._cond_trace: Optional[Trace] = None

    def get_trace(self) -> Trace:
        return Trace() if self._cond_trace is None else self._cond_trace

    def clear_cond_trace(self) -> None:
        self._cond_trace = None

@typechecked
class Traceable(Conditionable):
    def __init__(self) -> None:
        super().__init__()

    def log_probs(self, values:Optional[TraceLike] = None) -> Dict[str, Tensor]:
        #raise RuntimeError("traces are no longer cached for Observable and, therefore, all log_probs are invalid")
        logger.warn("traces are no longer CONSISTENTLY cached for Observable and, therefore, log_probs may be invalid")
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

        else:
            raise NotImplementedError()

    @property
    def variables(self) -> Set[str]:
        return set(self.get_trace().keys())


@typechecked
class UnusedLocalEffectHandler(ABC):
    def sample(self, dist:Any, value:Tensor, name:Optional[str]=None) -> None:
        raise NotImplementedError("requires different event handling structure.")


@typechecked
class Observable(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.observations:Dict[str, Tensor] = dict()

    def observe(self, key:str, value:Tensor, overwrite=False, warn=True) -> None:
        if key not in self.observations:
            self.observations[key] = value
        elif overwrite:
            if warn and not torch.equal(self.observations[key], value):
                logger.warning("overwrite used, but values differ")
            self.observations[key] = value


    def clear_conditions(self):
        self.observations = dict()

    def condition_on(self, obs:Trace, overwrite=False):
        for k, rv in obs.items():
            self.observe(k, rv.value, overwrite=overwrite)

    def update_conditions(self, conditions:Dict[str, Tensor], overwrite=False):
        """
        FIXME: this is almost same as condition_on: trace is an odict
        the main difference is that we use "update_conditions" on inference combinators
        """
        for k, v in conditions.items():
            self.observe(k, v, overwrite=overwrite)

    def _apply_observes(self, trace:Trace) -> Trace:
        for key, value in self.observations.items():
            trace.enqueue_observation(key, value)
        return trace

    def show_observations(self):
        if len(self.observations) == 0:
            print("No observations enqueued")
        else:
            print("Observations:")
            fill_size = max(map(lambda x: len(x), self.observations.keys())) + 2 # for quotes
            fill_template = "{:"+str(fill_size)+"}"
            key_template = lambda k: fill_template.format("'{}'".format(k))
            for k, v in self.observations.items():
                print("  {}: {}".format(key_template(k), tensor_utils.show(v)))


def with_observations(observations:TraceLike, runnable:Callable[[Trace], Output])->Output:
    """ A better function that summarizes Observable """
    # FIXME: replace the above with this
    trace = Trace()
    for key, v in observations.items():
        trace.enqueue_observation(key, get_value(v))
    return runnable(trace)

class TraceModule(Traceable, nn.Module): # , Observable):
    def __init__(self):
        # Observable.__init__(self)
        Traceable.__init__(self)
        nn.Module.__init__(self)
