#!/usr/bin/env python3

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
import math

from typing import NoReturn, Tuple

import operator
from itertools import accumulate
from functools import partial
from collections import namedtuple
from combinators.inference import _dispatch
from combinators import Forward, Reverse, Propose, Condition, Resample, RequiresGrad
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.tensor.utils import autodevice, kw_autodevice
from combinators.stochastic import Trace
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
import combinators.debug as debug
from combinators.utils import ppr, curry
import experiments.apgs_gmm.hao as hao
from combinators.trace.utils import RequiresGrad, copysubtrace, copytrace, mapvalues, disteq
from experiments.apgs_gmm.models import mk_target, Enc_rws_eta, Enc_apg_z, Enc_apg_eta, GenerativeOriginal, ix
from experiments.apgs_gmm.objectives import resample_variables, apg_update_z

from torch.distributions.one_hot_categorical import OneHotCategorical as cat

if debug.runtime() == 'jupyter':
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm

def printequal(desc, l, r, quiet=False, atol=1e-3, assertion=True):
    l, r = l.detach(), r.detach()
    eq = torch.equal(l, r)
    if eq:
        if quiet:
            return
        else:
            print(desc, "[ OK ]")
            return
    if not assertion:
        print(desc, "not equal")

    close = torch.allclose(l, r, atol=atol)
    if close:
        if not quiet:
            print(desc, "[WARNING] => close, not equal")
    else:
        if assertion:
            assert False, desc
        else:
            print(desc, "[ERROR] => not equal!!!!!")


def mk_metrics(loss, out, num_sweeps=1, ess_required=True, mode_required=True, density_required=True, mode_required_with_z=False):
    with torch.no_grad():
        metrics = dict()
        metrics['loss'] = loss.mean().cpu().item()
        if ess_required:
            w = F.softmax(out.log_prob.detach(), 0)
            metrics['ess'] = (1. / (w**2).sum(0)).mean().cpu().item()
        if density_required:
            metrics['density'] = out.target.log_prob.detach().mean().cpu().item()

        if num_sweeps > 1:
            pass
            # exc_kl, inc_kl= kls_eta(models, x, z_true)
            # if 'inc_kl' in metrics:
            #     metrics['inc_kl'] += inc_kl
            # else:
            #     metrics['inc_kl'] = inc_kl
            # if 'exc_kl' in metrics:
            #     metrics['exc_kl'] += exc_kl
            # else:
            #     metrics['exc_kl'] = exc_kl
        return metrics

def train(objective, models, target, data, assignments, num_epochs, sample_size, batch_size, normal_gamma_priors, with_tensorboard=False, lr=1e-3, seed=1, eval_break=50, is_smoketest=False, num_sweeps=None):
    # (num_clusters, data_dim, normal_gamma_priors, num_iterations=10000, num_samples = 600batches=50)
    """ data size  S * B * N * 2 """
    # Setup
    debug.seed(seed)
    # writer = debug.MaybeWriter(enable=with_tensorboard)
    loss_ct, loss_sum = 0, 0.0

    [enc_rws_eta, enc_apg_z, enc_apg_eta], generative = models, target

    assert all([len(list(k.parameters())) >  0 for k in models])
    optimizer = optim.Adam([dict(params=x.parameters()) for x in models], lr=lr)
    num_batches = int((data.shape[0] / batch_size))
    # sample_shape = (batch_size, sample_size)
    epochs = range(1) if is_smoketest else trange(num_epochs, desc='Epochs', position=1)
    for e in epochs:
        data, assignments = shuffler(data, assignments)

        num_batches = 3 if is_smoketest else num_batches
        with trange(num_batches, desc=f'Batch {{:{int(math.log(num_epochs, 10))}d}}'.format(e+1), position=0) as batches:
            for bix, b in enumerate(batches):
                optimizer.zero_grad()
                x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)

                loss, metrics = objective(enc_rws_eta=enc_rws_eta, enc_apg_z=enc_apg_z, enc_apg_eta=enc_apg_eta, generative=generative, x=x, sample_size=sample_size, num_sweeps=num_sweeps, compare=bix==0 or bix==num_batches-1)

                loss.backward()
                optimizer.step()

                # REPORTING
                # ---------------------------------------
                with torch.no_grad():
                    batches.set_postfix_str(";".join([f'{k}={{: .2f}}'.format(v) for k, v in metrics.items()])) # loss=loss.detach().cpu().mean().item(), **metrics)

                if is_smoketest and bix > 3:
                    return None


def shuffler(data, assignments):
    """
    shuffle the GMM datasets by both permuting the order of GMM instances (w.r.t. DIM1) and permuting the order of data points in each instance (w.r.t. DIM2)
    """
    concat_var = torch.cat((data, assignments), dim=-1)
    DIM1, DIM2, DIM3 = concat_var.shape
    indices_DIM1 = torch.randperm(DIM1)
    concat_var = concat_var[indices_DIM1]
    indices_DIM2 = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)], dim=0)
    concat_var = torch.gather(concat_var, 1, indices_DIM2.unsqueeze(-1).repeat(1, 1, DIM3))
    return concat_var[:,:,:2], concat_var[:,:,2:]


def main(is_smoketest, num_sweeps, simulate, objective):
    import subprocess
    gitroot = subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode("utf-8").rstrip()

    debug.seed(1)

    data_path=f'{gitroot}/data/gmm/'
    if simulate:
        from experiments.apgs_gmm.simulator import Sim_GMM
        simulator = Sim_GMM(N=60, K=3, D=2, alpha=2.0, beta=2.0, mu=0.0, nu=0.1)
        simulator.sim_save_data(num_seqs=10000, data_path=data_path)

    data = torch.from_numpy(np.load(f'{data_path}ob.npy')).float()
    assignments = torch.from_numpy(np.load(f'{data_path}assignment.npy')).float()

    # hyperparameters
    num_epochs=1 if is_smoketest else 100
    batch_size=50 if is_smoketest else 10
    budget=100
    lr=2e-4
    num_clusters=K=3
    data_dim=D=2
    num_hidden=30
    is_smoketest=debug.is_smoketest()

    normal_gamma_priors = dict(
        mu=torch.zeros((num_clusters, data_dim)),
        nu=torch.ones((num_clusters, data_dim)) * 0.1,
        alpha=torch.ones((num_clusters, data_dim)) * 2.0,
        beta=torch.ones((num_clusters, data_dim)) * 2.0,
    )
    # computable params
    num_batches = (data.shape[0] // batch_size)
    sample_size = budget // num_sweeps

    # Models
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden=num_hidden)
    enc_apg_eta = Enc_apg_eta(K, D)
    # generative = Generative(K, normal_gamma_priors)
    generative = GenerativeOriginal(K, D, False, 'cpu')

    config = dict(objective=objective, num_sweeps=num_sweeps, num_epochs=num_epochs, batch_size=batch_size, sample_size=sample_size, is_smoketest=is_smoketest)
    print(";".join([f'{k}={v}' for k, v in config.items()]))

    train(
        models=[enc_rws_eta, enc_apg_z, enc_apg_eta],
        target=generative,
        data=data,
        assignments=assignments,
        normal_gamma_priors=normal_gamma_priors,
        **config
    )
    print('tada!')

# ==================================================================================================================== #

def rws_objective_eager(enc_rws_eta, enc_apg_z, generative, og, x, enc_apg_eta=None, compare=True, sample_size=None, num_sweeps=1):
    """ One-shot for eta and z, like a normal RWS """
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    assert num_sweeps == 1
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out, _ = hao.oneshot(enc_rws_eta, enc_apg_z, og, x, metrics=metrics)
        debug.seed(1)

    # otherwise, eager combinators looks like:
    prp = Propose(proposal=Forward(enc_apg_z, enc_rws_eta), target=og, ix=ix(sweep=1,rev=False,block='is'))
    out = prp(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    loss = (w * (- out.proposal.log_prob)).sum(0).mean()
    if compare:
        assert torch.allclose(loss, _loss)
        assert torch.allclose(log_w, _log_w)
        assert trace_utils.valeq(out.trace, _q_eta_z_out.trace)

    return loss, mk_metrics(loss, out)

def rws_objective_declarative(enc_rws_eta, enc_apg_z, generative, og, x, enc_apg_eta=None, compare=True, sample_size=None, num_sweeps=1):
    assert num_sweeps == 1
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    if compare:
        debug.seed(1)
        _loss, _log_w, _q_eta_z_out, _  = hao.oneshot(enc_rws_eta, enc_apg_z, og, x, metrics=metrics)
        debug.seed(1)

    def loss_fn(out, loss):
        log_w = out.log_prob.detach()
        w = F.softmax(log_w, 0)
        return (w * (- out.proposal.log_prob)).sum(0).mean() + loss

    # otherwise, eager combinators looks like:
    prp = Propose(
        ix=ix(sweep=1,rev=False, block='is'),
        loss_fn=loss_fn,
        target=og,
        proposal=Forward(enc_apg_z, enc_rws_eta),
        )
    out = prp(x=x, prior_ng=og.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False)

    if compare:
        assert torch.allclose(out.loss, _loss)
        assert torch.allclose(out.log_prob, _log_w)
        assert trace_utils.valeq(out.trace, _q_eta_z_out.trace)

    return out.loss, mk_metrics(out.loss, out)

def test_rws_vae_eager():
    main(objective=rws_objective_eager, num_sweeps=1, is_smoketest=debug.is_smoketest(), simulate=False)

def test_rws_vae_declarative():
    main(objective=rws_objective_declarative, num_sweeps=1, is_smoketest=debug.is_smoketest(), simulate=False)


def apg_objective_eager2(enc_rws_eta, enc_apg_z, enc_apg_eta, generative, og, x, sample_size, num_sweeps, compare=True):
    assert num_sweeps == 2
    if compare:
        debug.seed(1)

        from combinators.resampling.strategies import APGSResamplerOriginal
        resampler = APGSResamplerOriginal(sample_size)
        sweeps, metrics = hao.apg_objective((enc_rws_eta, enc_apg_z, enc_apg_eta, og), x, num_sweeps, resampler)
        debug.seed(1)

    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    kwargs = dict(x=x, prior_ng=og.prior_ng, **jkwargs)
    prp1 = Propose(ix=ix(sweep=1,rev=False,block='is'),
            target=og, # og == original generator
            #          p(x,η_1,z_1)
            # ---------------------------------- [prp1]
            #          q(z_1 | η_1, x) q_0(η_1|x)
            proposal=Forward(enc_apg_z, enc_rws_eta))

    out1 = prp1(**kwargs, _debug=True)
    log_w1 = out1.log_cweight.detach()
    loss1 = (F.softmax(log_w1, 0) * (- out1.proposal.log_prob)).sum(0).mean()

    def printequal(desc, l, r, quiet=False, atol=1e-3, assertion=True):
        l, r = l.detach(), r.detach()
        eq = torch.equal(l, r)
        if eq:
            if quiet:
                return
            else:
                print(desc, "[ OK ]")
                return
        if not assertion:
            print(desc, "not equal")

        close = torch.allclose(l, r, atol=atol)
        if close:
            if not quiet:
                print(desc, "[WARNING] => close, not equal")
        else:
            if assertion:
                assert False, desc
            else:
                print(desc, "[ERROR] => not equal!!!!!")


    if compare:
        hao1 = sweeps[1]
        mas = lambda t: (t.detach().mean(), t.detach().sum())
        lj = lambda tr: mas(tr.log_joint(**jkwargs))
        print()
        printequal("sweep 1: target log_prob (log_p)", out1.target.log_prob, hao1['aux']['log_p'], quiet=True)
        printequal("sweep 1: propos log_prob (log_q)", out1.proposal.log_prob, hao1['aux']['log_q'], quiet=True)
        printequal("sweep 1: log_weight (log_w)     ", out1.log_cweight, hao1['log_w'], quiet=True)

        assert torch.equal(out1.target.log_prob, hao1['aux']['log_p'])
        assert torch.equal(out1.proposal.log_prob, hao1['aux']['log_q'])

        assert torch.equal(log_w1, hao1['log_w'])
        assert torch.equal(loss1, hao1['metrics']['iloss'][0])
        print("sweep1 log_w", log_w1.mean())
        assert trace_utils.valeq(out1.trace, hao1['q_eta_z'])
        debug.seed(1)
        print("=================================")
        print("===       IS cleared          ===")
        print("=================================")
        print(sweeps[2][1]['q_eta_z'])
        print()
        print(sweeps[2][1]['aux']['log_q_f'].sum(), sweeps[2][1]['aux']['log_q_f'].mean())
        assert set(out1.target.trace.keys()) == set(out1.trace.keys())
        dlist = list(set(out1.trace.keys()) - set(out1.proposal.trace.keys()))
        assert len(dlist) == 1 and dlist[0][:3] == 'lls'
        debug.seed(1)

    prp21 = Propose(
            target=Reverse(og, enc_apg_eta, ix=ix(sweep=2,rev=True,block='eta')), # , exclude={'lls'}),
            # target=Reverse(og, enc_apg_eta, ix=ix(sweep=2,rev=True,block='eta'), exclude={'lls'}),
            #          p(x,η_2,z) q(η_1 | z, x)
            # ---------------------------------------
            #          q(η_2 | z, x) prp_1(η_1,x,z)
            proposal=Forward(enc_apg_eta, prp1, ix=ix(sweep=2,rev=False,block='eta'))) # , exclude={'lls'}))
            # proposal=Forward(enc_apg_eta, prp1, ix=ix(sweep=2,rev=False,block='eta'), exclude={'lls'}))

    out21 = prp21(**kwargs, _debug=True, _debug_extras=out1)

    log_w21 = out21.log_weight.detach()
    w21 = F.softmax(log_w21, 0)
    log_q_fwd21_copied = trace_utils.copysubtrace(out21.proposal.kernel.trace, set(out21.proposal.trace.keys()) - set(out21.proposal.program.trace.keys())).log_joint(**jkwargs)
    loss21_copied = (w21 * (- log_q_fwd21_copied)).sum(0).mean()

    log_q_fwd21_computed = out21.proposal.log_prob - out21.proposal.program.log_prob
    loss21 = (w21 * (- log_q_fwd21_computed)).sum(0).mean()

    if compare:
        print("=================================")
        print("===     Sweep 2 Block Eta     ===")
        print("=================================")
        slog_q = sweeps[1]['aux']['log_q']
        log_q = out21.proposal.program.proposal.log_prob
        printequal("- sweep 1: log_q", slog_q.detach(), log_q.detach())

        slog_p = sweeps[1]['aux']['log_p']
        log_p = out21.proposal.program.target.log_prob
        printequal("- sweep 1: log_p", slog_p.detach(), log_p.detach())

        slog_w = sweeps[1]['log_w']
        log_w = out21.proposal.program.log_cweight
        printequal("- sweep 1: log_w", slog_w.detach(), log_w.detach())
        print("- IS step unchanged...     [ OK ]")
        eta2s = { 'means2', 'precisions2' }
        eta1s = { 'means1', 'precisions1' }
        lls = { 'lls' }
        z1s = { 'states1' }
        dset = set(out21.proposal.trace.keys()) - set(out21.proposal.program.trace.keys())
        assert dset == eta2s
        assert (eta2s | lls | z1s) == set(out21.target.program.trace.keys())
        assert (eta2s | eta1s | lls | z1s) == set(out21.target.kernel.trace.keys())
        assert (eta2s | lls | z1s) == set(out21.target.trace.keys())
        print("- traces are expected...   [ OK ]")

        # =============================================== #
        sweeps2eta = sweeps[2][1]
        hao21 = sweeps2eta['aux']
        # log_qp_fwd_sub = trace_utils.copysubtrace(out21.proposal.trace, eta2s | lls | z1s | eta1s).log_joint(**jkwargs)
        # print(log_qp_fwd_sub.mean())
        # slog_qp_b = hao21['log_q_f'] + hao21['log_p_b']
        # print(slog_qp_b.mean())
        # print(out21.proposal.log_prob.mean())
        # print(set(out21.proposal.trace.keys()) - (eta2s | lls | z1s | eta1s))
        proposal21 = out21.proposal
        target21   = out21.target
        printequal("- proposal log probs (program)...           ", hao21['log_p_b'], proposal21.program.log_prob)
        printequal("- proposal log probs (full)...              ", hao21['log_q_f'] + hao21['log_p_b'], proposal21.log_prob)
        printequal("- proposal log probs (kernel, algebraic)... ", hao21['log_q_f'], proposal21.log_prob - proposal21.program.log_prob)
        printequal("- proposal log probs (kernel, copied)...    ", hao21['log_q_f'], trace_utils.copysubtrace(proposal21.kernel.trace, set(proposal21.kernel.trace.keys()) - set(proposal21.program.trace.keys())).log_joint(**jkwargs))
        printequal("- target log probs (program)...             ", hao21['log_p_f'], target21.program.log_prob)
        printequal("- target log probs (full)...                ", hao21['log_p_f'] + hao21['log_q_b'], target21.log_prob)
        printequal("- target log probs (kernel, algebraic)...   ", hao21['log_q_b'], target21.log_prob - target21.program.log_prob)
        printequal("- target log probs (kernel, copied)...      ", hao21['log_q_b'], trace_utils.copysubtrace(target21.kernel.trace, set(target21.kernel.trace.keys()) - set(target21.program.trace.keys())).log_joint(**jkwargs))
        printequal("- incremental weight...                     ", hao21['log_w'], out21.log_weight)

        loss21_hao = sweeps[2][1]['metrics']['iloss'][1]
        printequal("- sweep 2 loss (kernel copied)...   ", loss21_hao, loss21_copied)
        printequal("- sweep 2 loss (kernel computed)... ", loss21_hao, loss21)
        debug.seed(1)

    prp22 = Propose(
            target=Reverse(og, enc_apg_z, ix=ix(sweep=2,rev=True, block='z')),
            #          p(x η_2 z_2) q( z_1 | η_2 x)
            # ---------------------------------------
            #          q(z_2 | η_2 x) prp_1(η_2 x z_1)
            proposal=Forward(enc_apg_z, prp21, ix=ix(sweep=2,rev=False, block='z')))

    out22 = prp22(**kwargs, _debug=True, _debug_extras=out1)
    log_w22 = out22.log_weight.detach()
    w22 = F.softmax(log_w22, 0)
    log_q_fwd22 = out22.proposal.log_prob - out22.proposal.program.trace.log_joint(**jkwargs)
    loss22 = (w22 * (- log_q_fwd22)).sum(0).sum(-1).mean()

    if compare:
        print("=================================")
        print("===     Sweep 2 Block Z       ===")
        print("=================================")
        proposal21 = out22.proposal.program.proposal
        target21   = out22.proposal.program.target
        printequal("- (block-eta) proposal log probs (program)...           ", hao21['log_p_b'],                    proposal21.program.log_prob)
        printequal("- (block-eta) proposal log probs (full)...              ", hao21['log_q_f'] + hao21['log_p_b'], proposal21.log_prob)
        printequal("- (block-eta) proposal log probs (kernel, algebraic)... ", hao21['log_q_f'],                    proposal21.log_prob - proposal21.program.log_prob)
        printequal("- (block-eta) proposal log probs (kernel, copied)...    ", hao21['log_q_f'],                    trace_utils.copysubtrace(proposal21.kernel.trace, set(proposal21.kernel.trace.keys()) - set(proposal21.program.trace.keys())).log_joint(**jkwargs))
        printequal("- (block-eta) target log probs (program)...             ", hao21['log_p_f'],                    target21.program.log_prob)
        printequal("- (block-eta) target log probs (full)...                ", hao21['log_p_f'] + hao21['log_q_b'], target21.log_prob)
        printequal("- (block-eta) target log probs (kernel, algebraic)...   ", hao21['log_q_b'],                    target21.log_prob - target21.program.log_prob)
        printequal("- (block-eta) target log probs (kernel, copied)...      ", hao21['log_q_b'],                    trace_utils.copysubtrace(target21.kernel.trace, set(target21.kernel.trace.keys()) - set(target21.program.trace.keys())).log_joint(**jkwargs))
        printequal("- (block-eta) incremental weight...                     ", sweeps2eta['log_w_eta'], out22.proposal.program.log_weight)

        sweeps2z   = sweeps[2][2]
        hao22 = sweeps2z['aux']
        proposal22 = out22.proposal
        target22   = out22.target
        printequal("- (block-z) proposal log probs (program, computed)...          ", hao22['log_p_b'],                    proposal22.program.trace.log_joint(**jkwargs))
        printequal("- (block-z) proposal log probs (full)...                       ", hao22['log_q_f'] + hao22['log_p_b'], proposal22.log_prob)
        printequal("- (block-z) proposal log probs (kernel, copied)...             ", hao22['log_q_f'],                    trace_utils.copysubtrace(proposal22.kernel.trace, set(proposal22.kernel.trace.keys()) - set(proposal22.program.trace.keys())).log_joint(**jkwargs))
        printequal("- (block-z) proposal log probs (kernel, computed+algebraic)... ", hao22['log_q_f'],                    proposal22.log_prob - proposal22.program.trace.log_joint(**jkwargs))

        printequal("- (block-z) target log probs (program)...             ", hao22['log_p_f'],                    target22.program.log_prob)
        printequal("- (block-z) target log probs (full)...                ", hao22['log_p_f'] + hao22['log_q_b'], target22.log_prob)
        printequal("- (block-z) target log probs (kernel, algebraic)...   ", hao22['log_q_b'],                    target22.log_prob - target22.program.log_prob)
        printequal("- (block-z) target log probs (kernel, copied)...      ", hao22['log_q_b'],                    trace_utils.copysubtrace(target22.kernel.trace, set(target22.kernel.trace.keys()) - set(target22.program.trace.keys())).log_joint(**jkwargs))
        printequal("- (block-z) incremental weight...                     ", sweeps2z['log_w_z'], out22.log_weight)
        printequal("- (block-z) loss...                                   ", sweeps2z['metrics']['iloss'][2], loss22)
        debug.seed(1)

    loss = loss1 + loss21 + loss22

    return loss, dict(loss1=loss1, loss21=loss21, loss22=loss22)

def test_apg_2sweep_eager():
    main(objective=apg_objective_eager2, num_sweeps=2, is_smoketest=debug.is_smoketest(), simulate=False)


def apg_objective_declarative(enc_rws_eta, enc_apg_z, enc_apg_eta, generative, x, sample_size, num_sweeps, compare=True):
    assert num_sweeps == 2
    if compare:
        debug.seed(1)
        from combinators.resampling.strategies import APGSResamplerOriginal
        resampler = APGSResamplerOriginal(sample_size)
        sweeps, metrics = hao.apg_objective((enc_rws_eta, enc_apg_z, enc_apg_eta, generative), x, num_sweeps, resampler)
        print(sweeps[1]['metrics']['iloss'])
        debug.seed(1)

    def loss_fn(cur, total_loss):
        ix = cur.proposal.ix
        jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)

        log_w = cur.log_weight.detach()
        w = F.softmax(log_w, 0)
        log_q = cur.proposal.log_prob if ix.block == "is" else \
            cur.proposal.log_prob - cur.proposal.program.trace.log_joint(**jkwargs)

        batch_loss = (w * (- log_q)).sum(0)
        if ix.block == 'z':
            batch_loss = batch_loss.sum(-1) # account for one-hot encoding
        return batch_loss.mean() + total_loss

    isix = ix(sweep=1,rev=False,block='is')
    propose = Propose(
        loss_fn=loss_fn,
        target=generative, ix=isix,
        #                   p(x,η_1,z_1)
        #        ---------------------------------------
        #            q(z_1 | η_1 x) q_rws(η_1 | x)
        proposal=Forward(enc_apg_z, enc_rws_eta, ix=isix))

    for sweep in range(2, num_sweeps+1):
        mkix = lambda block: (ix(sweep, True, block), ix(sweep, False, block))
        rix, fix = mkix('eta')
        propose_eta = Propose(loss_fn=loss_fn,
            target=Reverse(generative, enc_apg_eta, ix=rix),
            #                p(x η_2 z_1) q(η_1 | z_1 x)
            #           --------------------------------------
            #              q(η_2 | z_1 x) p(η_1,x,z_1)
            proposal=Forward(enc_apg_eta, propose, ix=fix))

        rix, fix = mkix('z')
        propose_z = Propose(loss_fn=loss_fn,
            target=Reverse(generative, enc_apg_z, ix=rix),
            #             p(x η_2 z_2) q( z_1 | η_2 x)
            #        ----------------------------------------
            #            q(z_2 | η_2 x) p(η_2 x z_1)
            proposal=Forward(enc_apg_z, propose_eta, ix=fix))

        propose = propose_z

    out = propose(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=compare)

    def traverse(out, getter):
        current = out
        returns = []
        while True:
            try:
                returns.append(getter(current))
            except:
                break
            if current.type != "Propose":
                break
            else:
                current = current.proposal.program

        returns.reverse()
        return returns

    losses = traverse(out, lambda p: p.loss.squeeze())
    if compare:
        losses_hao = list(accumulate(sweeps[1]['metrics']['iloss'], operator.add))
        assert len(losses) == len(losses_hao)
        mismatch_losses = list(filter(lambda xs: not torch.equal(*xs), zip(losses, losses_hao)))
        assert len(mismatch_losses) == 0


    return out.loss, {f"loss{i}":v.detach().cpu().item() for i, v in enumerate(losses)}

def test_apg_2sweep_declarative():
    main(objective=apg_objective_declarative, num_sweeps=2, is_smoketest=debug.is_smoketest(), simulate=False)
