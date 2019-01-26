#!/usr/bin/env python3

from collections import MutableMapping
from enum import Enum
import flatdict
import torch

from probtorch.stochastic import RandomVariable

import utils

class Provision(Enum):
    SAMPLED = 0
    OBSERVED = 1
    PROPOSED = 2

class HierarchicalTrace(MutableMapping):
    def __init__(self, proposal={}, proposal_slice=None):
        self._trie = flatdict.FlatDict(delimiter='/')
        self._var_list = []
        self._proposal = proposal
        if isinstance(proposal_slice, slice):
            self._proposal_slice = (proposal_slice.start, proposal_slice.stop)
        else:
            self._proposal_slice = (0, len(self._proposal))

    def __str__(self):
        return str(self._trie)

    def extract(self, prefix):
        if len(self._proposal):
            extracted_proposal = self._proposal.extract(prefix)
        else:
            extracted_proposal = {}
        subtrace = HierarchicalTrace(proposal=extracted_proposal)
        for k, v in self.get(prefix, {}).items():
            subtrace[k] = v
        return subtrace

    def insert(self, name, trace):
        for k in trace:
            self[name + '/' + k] = trace[k]

    def __setitem__(self, key, value):
        self._trie[key] = value
        self._var_list.append(key)

    def name(self, key):
        return self._var_list[key]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return HierarchicalTrace(proposal=self, proposal_slice=key)
        elif isinstance(key, int):
            return self._trie[self._var_list[key]]
        return self._trie[key]

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

        prov = Provision.SAMPLED
        if value is not None:
            if kwargs.get('proposed', False):
                prov = Provision.PROPOSED
            else:
                prov = Provision.OBSERVED
        if prov is Provision.SAMPLED:
            if self._proposal_slice[0] <= len(self) and len(self) < self._proposal_slice[1]:
                value = self._proposal.get(name, None)
                if value is not None:
                    value = value.value
                    prov = Provision.PROPOSED
                else:
                    prov = Provision.SAMPLED
        if prov is Provision.SAMPLED:
            value = utils.try_rsample(dist)

        self[name] = RandomVariable(dist, value, prov is Provision.OBSERVED)
        return value

    def sample(self, Dist, *args, **kwargs):
        assert 'value' not in kwargs
        return self.variable(Dist, *args, **kwargs)

    def param_sample(self, Dist, params, name):
        kwargs = {**params[name].copy(), 'name': name}
        for arg, val in kwargs.items():
            matches = [k for k in utils.PARAM_TRANSFORMS if k in arg]
            if matches:
                kwargs[arg] = utils.PARAM_TRANSFORMS[matches[0]](val)
        return self.sample(Dist, **kwargs)

    def observe(self, Dist, value, *args, **kwargs):
        assert 'value' not in kwargs
        kwargs['value'] = value
        return self.variable(Dist, *args, **kwargs)

    def param_observe(self, Dist, params, name, value):
        kwargs = {**params[name], 'name': name}
        for arg, val in kwargs.items():
            matches = [k for k in utils.PARAM_TRANSFORMS if k in arg]
            if matches:
                kwargs[arg] = utils.PARAM_TRANSFORMS[matches[0]](val)
        return self.observe(Dist, value, **kwargs)

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
        log_prob = torch.zeros(1).to(self.device)
        for n in nodes:
            if n in self._trie:
                node = self._trie[n]
                if isinstance(node, RandomVariable) and reparameterized and\
                   not node.reparameterized:
                    raise ValueError('All random variables must be sampled by reparameterization.')
                assert torch.isnan(node.log_prob).sum() == 0.0
                log_prob = utils.conjunct_events(log_prob, node.log_prob)
        return log_prob

    def log_weight(self):
        generative_joint = self.log_joint(reparameterized=False)
        latents = [rv for rv in self._trie if not self[rv].observed]
        unproposed = filter(lambda rv: self.proposed(rv) is None, latents)
        proposed = filter(lambda rv: self.proposed(rv) is not None, latents)
        proposed = list(proposed)

        if isinstance(self._proposal, HierarchicalTrace):
            unused_proposals = [var for var, rv in self._proposal.filter(
                lambda v, rv: not rv.observed and self.proposed(v) is not None
            )]
            unused_proposal_joint = self._proposal.log_joint(
                nodes=unused_proposals, reparameterized=False
            )
        else:
            unused_proposal_joint = torch.zeros(1).to(self.device)

        unproposed_joint = self.log_joint(nodes=unproposed,
                                          reparameterized=False)
        if proposed:
            proposed_joint = self._proposal.log_joint(nodes=proposed,
                                                      reparameterized=False)
        else:
            proposed_joint = torch.zeros(1).to(self.device)

        extended_generative_joint = generative_joint + unused_proposal_joint
        extended_guide_joint = unproposed_joint + proposed_joint

        return extended_generative_joint - extended_guide_joint

    def marginal_log_likelihood(self):
        weight = self.log_weight()
        if isinstance(weight, torch.Tensor):
            weight = utils.marginalize_all(weight)
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

    def find(self, predicate):
        for var in self:
            if predicate(var, self[var]):
                return (var, self[var])
