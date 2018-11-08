#!/usr/bin/env python3

from collections import MutableMapping
from enum import Enum
import flatdict
import torch

from probtorch.stochastic import RandomVariable
from probtorch.util import log_mean_exp

class Provision(Enum):
    SAMPLED = 0
    OBSERVED = 1
    PROPOSED = 2

class HierarchicalTrace(MutableMapping):
    def __init__(self, proposal={}, observations=lambda *args: None):
        self._trie = flatdict.FlatDict(delimiter='/')
        self._var_list = []
        self._proposal = proposal
        self._observations = observations

    def extract(self, prefix):
        if len(self._proposal):
            extracted_proposal = self._proposal.extract(prefix)
        else:
            extracted_proposal = {}
        subtrace = HierarchicalTrace(proposal=extracted_proposal,
                                     observations=self._observations)
        for k, v in self.get(prefix, {}).items():
            subtrace[k] = v
        return subtrace

    def insert(self, name, trace):
        for k in trace:
            self[name + '/' + k] = trace[k]

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

        prov = Provision.OBSERVED if value is not None else Provision.SAMPLED
        if prov is Provision.SAMPLED:
            value = self._observations(name, dist)
            if isinstance(value, RandomVariable):
                value = value.value
            prov = Provision.OBSERVED if value is not None\
                   else Provision.SAMPLED
        if prov is Provision.SAMPLED:
            value = self._proposal.get(name, None)
            if value is not None:
                value = value.value
                prov = Provision.PROPOSED
        if prov is Provision.SAMPLED:
            if dist.has_rsample:
                value = dist.rsample()
            else:
                value = dist.sample()

        self[name] = RandomVariable(dist, value, prov is Provision.OBSERVED)
        return value

    def sample(self, Dist, *args, **kwargs):
        assert 'value' not in kwargs
        return self.variable(Dist, *args, **kwargs)

    def param_sample(self, Dist, params, name):
        kwargs = {**params[name], 'name': name}
        return self.sample(Dist, **kwargs)

    def observe(self, Dist, value, *args, **kwargs):
        assert 'value' not in kwargs
        kwargs['value'] = value
        return self.variable(Dist, *args, **kwargs)

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

    @property
    def observations(self):
        return self._observations

    def log_joint(self, nodes=None, reparameterized=True):
        if nodes is None:
            nodes = list(self.keys())
        log_prob = torch.zeros(1).to(self.device)
        for n in nodes:
            if n in self._trie:
                node = self._trie[n]
                if isinstance(node, RandomVariable) and reparameterized and\
                   not node.reparameterized:
                    raise ValueError('All random variables must be sampled by reparameterization.')
                log_prob = log_prob + node.log_prob
        return log_prob

    def log_weight(self):
        generative_joint = self.log_joint(reparameterized=False)
        latents = [rv for rv in self._trie if not self[rv].observed]
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

    def filter(self, predicate):
        for var in self:
            if predicate(var, self[var]):
                yield (var, self[var])
