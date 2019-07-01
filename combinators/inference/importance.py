#!/usr/bin/env python3

from contextlib import contextmanager
import logging

from probtorch.stochastic import RandomVariable
import torch
import torch.distributions as dists

from ..sampler import Sampler
from . import inference
from ..model import foldable
from .. import utils

def conditioned_evaluate(target, xiq, log_wq, *args, **kwargs):
    with target.cond(xiq) as targetq:
        zs, xi, log_w = targetq(*args, **kwargs)
    log_omega_q = xiq.conditioning_factor(xi, target.batch_shape)
    return zs, xi, log_w + log_wq - log_omega_q

class Propose(inference.Inference):
    def __init__(self, target, proposal):
        super(Propose, self).__init__(target)
        assert isinstance(proposal, Sampler)
        assert proposal.batch_shape == target.batch_shape
        self.add_module('proposal', proposal)
        self._conditioned = False

    def forward(self, *args, **kwargs):
        if not self._conditioned:
            _, xiq, log_wq = self.proposal(*args, **kwargs)
            return conditioned_evaluate(self.target, xiq, log_wq, *args,
                                        **kwargs)
        return self.target(*args, **kwargs)

    def walk(self, f):
        return f(Propose(self.target.walk(f), self.proposal))

    @contextmanager
    def cond(self, qs):
        try:
            conditioned = self._conditioned
            self._conditioned = True
            yield super(Propose, self).cond(qs)
        finally:
            self._conditioned = conditioned

def propose(target, proposal):
    return Propose(target, proposal)

def collapsed_index_select(tensor, batch_shape, ancestors):
    tensor, unique = utils.batch_collapse(tensor, batch_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(batch_shape + unique)

def index_select_rv(rv, batch_shape, ancestors):
    result = rv
    if isinstance(rv, RandomVariable) and not rv.observed:
        value = collapsed_index_select(rv.value, batch_shape, ancestors)
        result = RandomVariable(rv.Dist, value, *rv.dist_args,
                                provenance=rv.provenance, mask=rv.mask,
                                **rv.dist_kwargs)
    return result

class Resample(inference.Inference):
    def forward(self, *args, **kwargs):
        zs, xi, log_weights = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        ancestors, log_weights = utils.gumbel_max_resample(log_weights)

        zs = list(zs)
        for i, z in enumerate(zs):
            if isinstance(z, torch.Tensor):
                zs[i] = collapsed_index_select(z, self.batch_shape, ancestors)
        if multiple_zs:
            zs = tuple(zs)
        else:
            zs = zs[0]

        resampler = lambda rv: index_select_rv(rv, self.batch_shape, ancestors)
        trace_resampler = lambda _, trace: utils.trace_map(trace, resampler)
        xi = xi.map(trace_resampler)

        return zs, xi, log_weights

    def walk(self, f):
        return f(Resample(self.target.walk(f)))

def resample(target):
    return Resample(target)

def resample_proposed(target, proposal):
    return resample(propose(target, proposal))

def smc(target):
    selector = lambda m: isinstance(m, Propose)
    return target.apply(resample, selector)

def step_smc(target):
    selector = lambda m: isinstance(m, foldable.Step)
    return target.apply(resample, selector)

def dreg(log_weight, alpha=torch.zeros(())):
    probs = utils.normalize_weights(log_weight).detach().exp()
    particles = (alpha * probs + (1 - 2 * alpha) * probs**2) * log_weight
    return utils.batch_sum(particles)

def elbo(log_weight, iwae_objective=False, xi=None):
    if xi and xi.reparameterized() and iwae_objective:
        return dreg(log_weight, alpha=torch.zeros(()))
    elif iwae_objective:
        return utils.batch_marginalize(log_weight)
    return utils.batch_mean(log_weight)

def eubo(log_weight, iwae_objective=False, xi=None, inference_params=True):
    sign = -1.0 if inference_params else 1.0
    if xi and xi.reparameterized():
        return sign * dreg(log_weight, alpha=torch.ones(()))
    else:
        probs = utils.normalize_weights(log_weight).detach().exp()
        eubo_particles = probs * log_weight
        if iwae_objective:
            return sign * utils.log_sum_exp(eubo_particles)
        return sign * utils.batch_sum(eubo_particles)

class EvBoOptimizer:
    def __init__(self, param_groups, optimizer_constructor):
        self._num_groups = len(param_groups)
        self._objectives = [g['objective'] for g in param_groups]
        self._optimizers = [optimizer_constructor([g['optimizer_args']])
                            for g in param_groups]
        self._schedules = [None for g in param_groups]
        for g, group in enumerate(param_groups):
            if 'patience' in group and group['patience'] is not None:
                self._schedules[g] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizers[g], factor=0.5, min_lr=1e-6,
                    patience=group['patience'], verbose=True, mode='min',
                )

    def zero_grads(self):
        for optimizer in self._optimizers:
            optimizer.zero_grad()

    def step_grads(self, log_weight, xi):
        objectives = []
        for g in range(self._num_groups):
            objective = self._objectives[g]['function'](log_weight, xi=xi)
            objectives.append(objective)
        for g in range(self._num_groups):
            if g == self._num_groups - 1:
                objectives[g].backward()
            else:
                objectives[g].backward(retain_graph=True)
            self._optimizers[g].step()
            if self._schedules[g]:
                self._schedules[g].step(objective)
        return [objective.detach().cpu() for objective in objectives]

def default_elbo_logger(objectives, t, xi=None):
    logging.info('ELBO=%.8e at epoch %d', -objectives[0], t + 1)
    return [-objectives[0]]

def default_eubo_logger(objectives, t, xi=None):
    logging.info('EUBO=%.8e at epoch %d', objectives[1], t + 1)
    return [objectives[1]]

def multiobjective_variational(sampler, param_groups, num_iterations, data,
                               use_cuda=True, logger=default_elbo_logger):
    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    evbo_optim = EvBoOptimizer(param_groups, torch.optim.Adam)
    iteration_bounds = list(range(num_iterations))
    for i in range(num_iterations):
        evbo_optim.zero_grads()
        _, xi, log_weight = sampler(data=data)
        iteration_bounds[i] = evbo_optim.step_grads(log_weight, xi)
        if logger is not None:
            iteration_bounds[i] = logger(iteration_bounds[i], i, xi)

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
        torch.cuda.empty_cache()
    sampler.eval()

    trained_params = sampler.args_vardict(False)

    iteration_bounds = torch.stack([
        torch.stack(bounds, dim=-1) for bounds in iteration_bounds
    ], dim=0)
    return xi, trained_params, iteration_bounds

def variational_importance(sampler, num_iterations, data, use_cuda=True,
                           lr=1e-6, bound='elbo', log_all_bounds=False,
                           patience=50, log_estimator=False):
    sampler.train()
    if torch.cuda.is_available() and use_cuda:
        sampler.cuda()

    if bound == 'elbo':
        objective = {
            'name': bound,
            'function': lambda lw, xi=None: -elbo(lw,
                                                  iwae_objective=log_estimator,
                                                  xi=xi),
        }
    elif bound == 'eubo':
        objective = {
            'name': bound,
            'function': lambda lw, xi=None: eubo(lw,
                                                 iwae_objective=log_estimator,
                                                 xi=xi),
        }
    evbo_optim = EvBoOptimizer([{
        'objective': objective, 'patience': patience,
        'optimizer_args': {'params': list(sampler.parameters()), 'lr': lr},
    }], torch.optim.Adam)

    bounds = list(range(num_iterations))
    for t in range(num_iterations):
        evbo_optim.zero_grads()

        _, xi, log_weight = sampler(data=data)

        eubo_t = eubo(log_weight, iwae_objective=log_estimator, xi=xi)
        elbo_t = elbo(log_weight, iwae_objective=log_estimator, xi=xi)
        bounds[t] = {'eubo': eubo_t.detach(), 'elbo': elbo_t.detach()}
        if log_all_bounds:
            logging.info('ELBO=%.8e at epoch %d', elbo_t, t + 1)
            logging.info('EUBO=%.8e at epoch %d', eubo_t, t + 1)
        else:
            logging.info('%s=%.8e at epoch %d', bound.upper(), bounds[t][bound],
                         t + 1)

        evbo_optim.step_grads(log_weight, xi)

    if torch.cuda.is_available() and use_cuda:
        sampler.cpu()
        torch.cuda.empty_cache()
    sampler.eval()

    trained_params = sampler.args_vardict(False)
    bounds = (torch.stack([bs['elbo'] for bs in bounds], dim=0).detach().cpu(),
              torch.stack([bs['eubo'] for bs in bounds], dim=0).detach().cpu())

    return xi, trained_params, bounds
