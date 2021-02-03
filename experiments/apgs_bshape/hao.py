from functools import partial
from combinators.inference import Forward, Propose, Reverse
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import combinators.stochastic as probtorch
import combinators.debug as debug
from combinators.utils import ppr
import combinators.tensor.utils as tensor_utils
from experiments.apgs_bshape.models import Noop, ix, NoopKernel
from combinators.inference import *
import sys

def resample_variables(resampler, q, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    q_new = probtorch.Trace()
    for key, node in q.items():
        resampled_loc = resampler.resample_4dims(var=node.dist.loc, ancestral_index=ancestral_index)
        resampled_scale = resampler.resample_4dims(var=node.dist.scale, ancestral_index=ancestral_index)
        resampled_value = resampler.resample_4dims(var=node.value, ancestral_index=ancestral_index)
        q_new.normal(loc=resampled_loc, scale=resampled_scale, value=resampled_value, name=key)
    return q_new

def apg_objective(models, AT1, AT2, frames, K, result_flags, num_sweeps, resampler, mean_shape):
    """
    Amortized Population Gibbs objective in Bouncing Shapes problem
    """
    metrics = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_where' : [], 'E_recon' : [], 'density' : []}
    log_w, q, metrics, propose_IS = oneshot(models, frames, mean_shape, metrics, result_flags)
    # q = resample_variables(resampler, q, log_weights=log_w)
    # propose = Resample(propose_IS)
    propose = propose_IS
    T = frames.shape[2]
    (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]

    for m in range(num_sweeps-1):
        for t in range(T):
            log_w, q, metrics = apg_where_t(models, frames, q, t, metrics, result_flags)
            q = resample_variables(resampler, q, log_weights=log_w)
        log_w, q, metrics = apg_what(models, frames, q, metrics, result_flags)
        q = resample_variables(resampler, q, log_weights=log_w)

    for sweep in range(num_sweeps-1):
        mkix = lambda block: (ix(sweep, True, block, t=sweep+T+1, recon_level="unknown"), ix(sweep, False, block, t=sweep+T+1, recon_level="unknown"))

        rix, fix = mkix('where')
        for t in range(T):
            propose = Propose(
                target=Reverse(dec_coor2, enc_coor2, ix=rix),
                proposal=Forward(enc_coor2, propose, ix=fix))
            propose = Resample(propose)

        rix, fix = mkix('what')
        propose = Propose(
            target=Reverse(dec_digit2, enc_digit2, ix=rix),
            proposal=Forward(enc_digit2, propose, ix=rix),
        )
        propose = Resample(propose)

    # out = propose(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=compare)

    if result_flags['loss_required']:
        metrics['loss_phi'] = torch.cat(metrics['loss_phi'], 0)
        metrics['loss_theta'] = torch.cat(metrics['loss_theta'], 0)
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0)
    if result_flags['mode_required']:
        metrics['E_where'] = torch.cat(metrics['E_where'], 0)
        metrics['E_recon'] = torch.cat(metrics['E_recon'], 0)
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0)
    return metrics

def oneshot(models, frames, conv_kernel, metrics, result_flags):
    (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
    (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]
    T = frames.shape[2]
    S, B, K, DP, DP = conv_kernel.shape
    q = probtorch.Trace()
    p = probtorch.Trace()
    print("========================================================")

    log_q = torch.zeros(S, B)
    for t in range(T):
        q = enc_coor(q, frames, t, conv_kernel, extend_dir='forward')
        p = dec_coor(q, p, t)
        log_q += q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        log_w = (log_p - log_q)

        if t <= IX:
            ppr(log_q, desc="log_q ({})\t{: .4f}".format(t, log_q.mean().item()))
            ppr(log_p, desc="log_p ({})\t{: .4f}".format(t, log_p.mean().item()))
            ppr(log_w, desc="log_w ({})\t{: .4f}".format(t, log_w.mean().item()))
        if t <= IX:
            ppr(q, desc="hao proposal trace")
            ppr(p, desc="hao target trace")
            print()

    print("--------------------------------------------------------")
    # print("proposal trace:")
    # ppr(q, m='vd')
    # print("target trace:")
    # ppr(p, m='vd')

    mkix = lambda t: ix(sweep=1, rev=False, block='is', t=t, recon_level='frames')
    noopp = Noop()
    propose_IS = noopp
    for t in range(T):
        propose_IS = Propose(
            ix=mkix(t),
            target=dec_coor2,
            #       p_coor(c_1_t=0) ... p_coor(c_1_t=10)
            # ------------------------------------------------------
            #            q(z_1 | η_1, x) q_0(η_1|x)
            proposal=Forward(enc_coor2, propose_IS))

    out = propose_IS(dict(frames=frames, conv_kernel=conv_kernel, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)
    ppr(out.trace, m='v')


    # forward kernel shape
    q = enc_digit(q, frames, timestep=T+1, extend_dir='forward')
    # generative program shape
    p = dec_digit(q, p, frames, timestep=T+1, recon_level='frames')

    # print("POST: proposal trace:")
    # ppr(q, m='vd')
    # print("POST: target trace:")
    # ppr(p, m='vd')

    propose_IS = Propose(ix=mkix(T+1),
        target=dec_digit2,
        proposal=Forward(enc_digit2, propose_IS))

    out = propose_IS(dict(frames=frames, conv_kernel=conv_kernel, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)


    print('--------------------')
    log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    ppr(log_q, desc="log_q (hao)")
    print('--------------------')
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    ppr(out.target.log_prob, desc='log_p (com)')
    ppr(log_p,                 desc="log_p (hao)")
    print('--------------------')
    log_w = (log_p - log_q).detach()
    ppr(out.log_weight, desc='log_w (com)')
    ppr(log_w,          desc="log_w (hao)")
    print('--------------------')
    w = F.softmax(log_w, 0).detach()
    print('onestep done')

    if result_flags['loss_required']:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (-log_p)).sum(0).mean()
        metrics['loss_phi'].append(loss_phi.unsqueeze(0))
        metrics['loss_theta'].append(loss_theta.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. /(w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_where = []
        for t in range(T):
            E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
        E_where = torch.cat(z_where, 2)
        metrics['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * K * 2
        metrics['E_recon'].append(p['recon'].dist.probs.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * FP * FP
    if result_flags['density_required']:
        metrics['density'].append(log_p.detach().unsqueeze(0))
    return log_w, q, metrics, propose_IS

def apg_where_t(models, frames, q, timestep, metrics, result_flags):
    T = frames.shape[2]
    (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
    conv_kernel = dec_digit(q=q, p=None, frames=frames, timestep=timestep, recon_level='object')
    # forward
    q_f = enc_coor(q, frames, timestep, conv_kernel, extend_dir='forward')
    p_f = probtorch.Trace()
    p_f = dec_coor(q_f, p_f, timestep)
    if timestep < (T-1):
        p_f = dec_coor(q_f, p_f, timestep+1)
    if timestep > 0:
        p_f = dec_coor(q_f, p_f, timestep-1)
    p_f = dec_digit(q_f, p_f, frames[:,:,timestep,:,:], recon_level='frame', timestep=timestep)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_f = q_f['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_f = log_p_f - log_q_f
    # backward
    q_b = enc_coor(q, frames, timestep, conv_kernel, extend_dir='backward')
    p_b = probtorch.Trace()
    p_b = dec_coor(q_b, p_b, timestep)
    if timestep < (T-1):
        p_b = dec_coor(q_b, p_b, timestep+1)
    if timestep > 0:
        p_b = dec_coor(q_b, p_b, timestep-1)
    p_b = dec_digit(q_b, p_b, frames[:,:,timestep,:,:], recon_level='frame', timestep=timestep)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_b = q_b['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        metrics['loss_phi'].append((w * (- log_q_f)).sum(0).mean().unsqueeze(0))
        metrics['loss_theta'].append((w * (- log_p_f)).sum(0).mean().unsqueeze(0))
#     if result_flags['density_required']:
#         trace['density'].append(log_prior.unsqueeze(0).detach())
    return log_w, q_f, metrics


def apg_what(models, frames, q, metrics, result_flags):
    T = frames.shape[2]
    (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
    q_f = enc_digit(q, frames, timestep=T+1, extend_dir='forward')
    p_f = probtorch.Trace()
    p_f = dec_digit(q_f, p_f, frames, timestep=T+1, recon_level='frames')
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_f = q_f['z_what'].log_prob.sum(-1).sum(-1)
    log_w_f = log_p_f - log_q_f
    q_b = enc_digit(q, frames, timestep=T+1, extend_dir='backward')
    p_b = probtorch.Trace()
    p_b = dec_digit(q_b, p_b, frames, timestep=T+1, recon_level='frames')
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_b = q_b['z_what'].log_prob.sum(-1).sum(-1)
    log_w_b = log_p_b - log_q_b

    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w * (-log_q_f)).sum(0).mean()
        loss_theta = (w * (-log_p_f)).sum(0).mean()
        metrics['loss_phi'][-1] = metrics['loss_phi'][-1] + loss_phi.unsqueeze(0)
        metrics['loss_theta'][-1] = metrics['loss_theta'][-1] + loss_theta.unsqueeze(0)
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_where = []
        for t in range(T):
            E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
        E_where = torch.cat(z_where, 2)
        metrics['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach())
        metrics['E_recon'].append(p['recon'].dist.probs.mean(0).unsqueeze(0).detach())
    if result_flags['density_required']:
        log_joint = log_p_f.detach()
        for t in range(T):
            p_f = dec_coor(q_f, p_f, t)
        metrics['density'].append(p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False).unsqueeze(0).detach())
    return log_w, q_f, metrics


