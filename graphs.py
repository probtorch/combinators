#!/usr/bin/env python3

from functools import reduce
import probtorch
import pygtrie
import torch

import utils

class ModelGraph:
    def __init__(self, trie=None, traces=None):
        self._trie = trie if trie else pygtrie.StringTrie()
        self._ordering = list(trie.keys()) if trie else []
        if traces:
            for (name, trace) in traces.items():
                self[name] = trace

    def __setitem__(self, key, val):
        assert isinstance(val, probtorch.Trace)
        assert key not in self._trie and key not in self._ordering
        self._trie[key] = val
        self._ordering.append(key)

    def name(self, key):
        return self._ordering[key]

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def contains_model(self, prefix=''):
        return self._trie.has_subtrie(prefix)

    def __getitem__(self, key):
        if isinstance(key, slice):
            trie = pygtrie.StringTrie()
            key = key.start
            assert key[-1] != '/'
            for k, v in self._trie.iteritems(prefix=key):
                trie[k[len(key)+1:]] = v
            return ModelGraph(trie)
        elif isinstance(key, int):
            return self._trie[self._ordering[key]]
        assert not key or key[-1] != '/'
        return self._trie[key]

    def __delitem__(self, name):
        raise NotImplementedError('Cannot delete trace from a trace tree')

    def __len__(self):
        return len(self._trie)

    def __iter__(self):
        return self.keys()

    def __repr__(self):
        return 'ModelGraph{%s}' % str(self._trie.items())

    def keys(self):
        for address in self._ordering:
            yield address

    def values(self):
        for address in self._ordering:
            yield self[address]

    def items(self):
        for address in self._ordering:
            yield (address, self[address])

    def map(self, f):
        result = ModelGraph()
        for address in self:
            result[address] = f(address, self[address])
        return result

    def filter(self, predicate):
        for address in self:
            if predicate(address, self[address]):
                yield (address, self[address])

    def find(self, predicate):
        for address in self:
            if predicate(address, self[address]):
                return (address, self[address])

    @property
    def device(self):
        for addr in self:
            for v in self[addr]:
                return self[addr][v].value.device
        return 'cpu'

    def log_joint(self, prefix='', nodes=None):
        if nodes is None:
            hdr = pygtrie._SENTINEL if not prefix else prefix
            nodes = list(reduce(lambda x, y: x + y, [
                [addr + '/' + var for var in self._trie[addr]]
                for addr in utils.iter_trie_slice(self._trie, hdr)
            ]))
        log_prob = torch.zeros(1).to(self.device)
        for n in nodes:
            addr, _, var = n.rpartition('/')
            assert addr.startswith(prefix)
            node = self._trie[addr][var]
            assert torch.isnan(node.log_prob).sum() == 0.0
            log_prob = utils.conjunct_events(log_prob, node.log_prob)
        return log_prob

    def __mul__(self, other):
        result = ModelGraph()
        for k, v in self.items():
            result[k] = v
        for k, v in other.items():
            result[k] = v
        return result

    def __imul__(self, other):
        for k, v in other.items():
            self[k] = v
        return self

    def insert(self, prefix, other):
        assert isinstance(other, ModelGraph)
        for k, v in other.items():
            self[prefix + '/' + k] = v

    def variables(self, prefix=pygtrie._SENTINEL, predicate=lambda k, v: True):
        for _, trace in self._trie.iteritems(prefix=prefix):
            for k, v in trace.items():
                if predicate(k, v):
                    yield (k, v)

    def graft(self, key, val):
        if isinstance(key, int):
            key = self._ordering[key]
        assert isinstance(val, probtorch.Trace)
        result = ModelGraph(trie=self._trie.copy())
        result._ordering = self._ordering
        for i, k in enumerate(result._ordering):
            if k == key:
                result._ordering[i] = key
                break
        result._trie[key] = val
        return result
