#!/usr/bin/env python3

from functools import reduce
import probtorch
import pygtrie
import torch

from . import utils

class ComputationGraph:
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
            return ComputationGraph(trie)
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
        return 'ComputationGraph{%s}' % str(self._trie.items())

    def keys(self):
        for key in self._ordering:
            yield key

    def values(self):
        for key in self._ordering:
            yield self[key]

    def items(self):
        for key in self._ordering:
            yield (key, self[key])

    def map(self, f):
        result = ComputationGraph()
        for key in self:
            result[key] = f(key, self[key])
        return result

    def filter(self, predicate):
        for key in self:
            if predicate(key, self[key]):
                yield (key, self[key])

    def find(self, predicate):
        for key in self:
            if predicate(key, self[key]):
                return (key, self[key])
        return None

    @property
    def device(self):
        for key in self:
            for v in self[key]:
                return self[key][v].value.device
        return 'cpu'

    def log_joint(self, prefix='', nodes=None):
        if nodes is None:
            hdr = pygtrie._SENTINEL if not prefix else prefix
            nodes = list(reduce(lambda x, y: x + y, [
                [key + '/' + var for var in self._trie[key]]
                for key in utils.iter_trie_slice(self._trie, hdr)
            ]))
        log_prob = torch.zeros(1).to(self.device)
        for n in nodes:
            key, _, var = n.rpartition('/')
            assert key.startswith(prefix)
            node = self._trie[key][var]
            assert torch.isnan(node.log_prob).sum() == 0.0
            log_prob = utils.conjunct_events(log_prob, node.log_prob)
        return log_prob

    def __mul__(self, other):
        result = ComputationGraph()
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
        assert isinstance(other, ComputationGraph)
        for k, v in other.items():
            self[prefix + '/' + k] = v

    def variables(self, prefix=pygtrie._SENTINEL, predicate=lambda k, v: True):
        for _, trace in self._trie.iteritems(prefix=prefix):
            for k in trace.variables():
                v = trace[k]
                if predicate(k, v):
                    yield (k, v)

    def num_variables(self, prefix=pygtrie._SENTINEL,
                      predicate=lambda k, v: True):
        return len(list(self.variables(prefix, predicate)))

    def graft(self, key, val):
        if isinstance(key, int):
            key = self._ordering[key]
        assert isinstance(val, probtorch.Trace)
        result = ComputationGraph(trie=self._trie.copy())
        result._ordering = self._ordering
        for i, k in enumerate(result._ordering):
            if k == key:
                result._ordering[i] = key
                break
        result._trie[key] = val
        return result

    def markov_blanket(self, key):
        if isinstance(key, int):
            key = self._ordering[key]
        blanket = set()

        # Walk to the parent, add it and the siblings
        parent_key = key.rpartition('/')[0]
        if parent_key and self._trie.has_node(parent_key):
            blanket.add(parent_key)
            parent_depth = len(parent_key.split('/'))
            for k in self._trie.keys(prefix=parent_key):
                if len(k.split('/')) == parent_depth + 1 and k != key:
                    blanket.add(k)

        # Add children of this node
        key_depth = len(key.split('/'))
        for k in self._trie.keys(prefix=key):
            if len(k.split('/')) == key_depth + 1:
                blanket.add(k)

        return blanket
