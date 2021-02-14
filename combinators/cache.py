"""
module that defines generic property-based dicts (called "caches")
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple, Optional, Dict, List, Union, Set, Callable
from collections import ChainMap, namedtuple
from typeguard import typechecked
from abc import ABC, abstractmethod
import inspect
import ast
import weakref
from typing import Iterable

class Cache(ABC):
    def __init__(self, properties:List[str], *args, **kwargs):
        assert len(args) == 0 or len(args) == len(properties)
        assert len(kwargs) == 0 or (len(kwargs) == len(properties) and all([k in properties for k in kwargs.keys()]))
        # init
        _ = [setattr(self, prop, None) for prop in properties]
        # set args
        if len(args) == 0 and len(kwargs) == 0:
            _ = [setattr(self, prop, None) for prop in properties]
        elif len(args) > 0:
            _ = [setattr(self, k, v) for k, v in zip(properties, args)]
        elif len(kwargs) > 0:
            _ = [setattr(self, k, v) for k, v in kwargs.items()]
        else:
            raise TypeError()

        self._properties = properties
        fill_width = max(map(len, properties))
        self._property_template = "{:>" +str(fill_width)+ "}: {}"

    @property
    @abstractmethod
    def _cache_name(self):
        raise NotImplementedError()

    def __repr__(self):
        header = f"{self._cache_name} Cache"
        property_str = lambda prop: self._property_template.format(prop, getattr(self, prop))
        return "\n  ".join([f"{self._cache_name} Cache"] + [property_str(prop) for prop in self._properties])

class KCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(['program', 'kernel'], *args, **kwargs)

    @property
    def _cache_name(self):
        return "Kernel"

class PCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(['proposal', 'target'], *args, **kwargs)

    @property
    def _cache_name(self):
        return "Propose"
