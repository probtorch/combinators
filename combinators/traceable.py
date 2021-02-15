import logging
from typing import Optional
from typeguard import typechecked
from abc import ABC

from combinators.stochastic import Trace
import warnings
import functools

logger = logging.getLogger(__name__)

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

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

