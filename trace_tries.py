#!/usr/bin/env python3

from collections import MutableMapping
import flatdict

import numpy as np
import probtorch
from probtorch.stochastic import RandomVariable
from probtorch.util import batch_sum, log_mean_exp
import torch
import torch.nn as nn

import utils

class HierarchicalTrace(MutableMapping):
    def __init__(self, proposal={}):
        self._trie = flatdict.FlatDict(delimiter='/')
        self._var_list = []
        self._proposal = proposal

    def extract(self, prefix):
        result = HierarchicalTrace()
        for k, v in self[prefix].items():
            result[k] = v
        return result

    def insert(self, name, trace):
        for k, v in trace.variables():
            self[name + '/' + k] = v

    def __setitem__(self, key, value):
        self._trie[key] = value
        self._var_list.append(key)

    def __getitem__(self, name):
        return self._trie[name]

    def __delitem__(self, name):
        raise NotImplementedError('Cannot delete item from a trace')

    def __len__(self):
        return len(self._var_list)

    def __iter__(self):
        for var in self._var_list:
            yield var

    def variable(self, Dist, *args, **kwargs):
        """Creates a new RandomVariable node"""
        name = kwargs.pop('name', str(len(self._var_list)))
        value = kwargs.pop('value', None)
        dist = Dist(*args, **kwargs)
        if value is None:
            value = self._proposal.get(name, None)
            if value is None:
                if dist.has_rsample:
                    value = dist.rsample()
                else:
                    value = dist.sample()
            observed = False
        else:
            observed = True
            if isinstance(value, RandomVariable):
                value = value.value
        self[name] = RandomVariable(dist, value, observed)
        return value

    def proposed(self, name):
        rv = self._proposal.get(name, None)
        if rv is not None:
            rv = rv.value
        return rv

    @property
    def device(self):
        for v in self:
            return self[v].value.device
        return 'cpu'

    def log_joint(self, nodes=None, reparameterized=True):
        if nodes is None:
            nodes = list(self.keys())
        log_prob = torch.zeros().to(self.device)
        for n in nodes:
            if n in self._trie:
                node = self._trie[n]
                if isinstance(node, RandomVariable) and reparameterized and\
                   not node.reparameterized:
                    raise ValueError('All random variables must be sampled by reparameterization.')
                log_p = batch_sum(node.log_prob, None, None)
                log_prob = log_prob + log_p
        return log_prob

    def log_weight(self):
        generative_joint = self.log_joint(reparameterized=False)
        latents = [rv for rv in self._trie if not rv.observed]
        unproposed = filter(lambda rv: self.proposed(rv) is None, latents)
        proposed = filter(lambda rv: self.proposed(rv) is not None, latents)

        unproposed_joint = self.log_joint(nodes=unproposed,
                                          reparameterized=False)
        proposed_joint = self.log_joint(nodes=proposed, reparameterized=False)
        return generative_joint - (unproposed_joint + proposed_joint)

    def marginal_log_likelihood(self):
        weight = self.log_weight()
        if isinstance(weight, torch.Tensor):
            for _ in range(len(weight.shape)):
                weight = log_mean_exp(weight, dim=0)
        return weight

    def normalized_log_weight(self):
        return self.log_weight() - self.marginal_log_likelihood()

    def effective_sample_size(self):
        return (self.log_weight()*2).exp().sum(dim=0).pow(-1)

    def map(self, f):
        result = HierarchicalTrace(proposal=self._proposal)
        for var in self:
            result[var] = f(var, self[var])
        return result
