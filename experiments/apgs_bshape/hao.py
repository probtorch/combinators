from functools import partial
from combinators.inference import Forward, Propose, Reverse
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import combinators.stochastic as probtorch
import combinators.debug as debug
from combinators.utils import ppr, pprm
import combinators.tensor.utils as tensor_utils
from experiments.apgs_bshape.models import Noop, ix, Memo
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

def _get_conv(mean_shape, dec_digit):
    def get_conv(ix, trace=None, eval_output=None):
        if ix.block=='is':
            return mean_shape
        else:
            eval_output['reconstruction_trace']=trace
            out = dec_digit(ix=ix, eval_output=eval_output, shared_args=None)
            return out.output['conv_kernel']
    return get_conv

def apg_objective(models, AT1, AT2, frames, K, result_flags, num_sweeps, resampler, mean_shape):
    """
    Amortized Population Gibbs objective in Bouncing Shapes problem
    """
    metrics = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_where' : [], 'E_recon' : [], 'density' : []}
    (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]
    log_w, q, metrics, propose_IS = oneshot(models, frames, mean_shape, metrics, result_flags)

    # q = resample_variables(resampler, q, log_weights=log_w)

    T = frames.shape[2]

    for m in range(num_sweeps-1):
        for t in range(T):
            log_w, q, metrics = apg_where_t(models, frames, q, t, metrics, result_flags)
            q = resample_variables(resampler, q, log_weights=log_w)
        log_w, q, metrics = apg_what(models, frames, q, metrics, result_flags)
        q = resample_variables(resampler, q, log_weights=log_w)

    print("========================================================")
    get_conv = _get_conv(mean_shape, dec_digit2)

    mkix = lambda t: ix(sweep=1, rev=False, block='is', t=t, recon_level='frames')

    propose_IS = Noop()

    for t in range(T):
        proposal_IS = Forward(enc_coor2, propose_IS)

    propose_IS = Propose(
        ix=mkix(t),
        target=dec_coor2,
        proposal=propose_IS)

    propose_IS = Propose(ix=mkix(T+1),
        target=dec_digit2,
        proposal=Forward(enc_digit2, Resample(propose_IS)))

    propose = propose_IS
    out = propose(dict(frames=frames, get_conv=get_conv, conv_kernel=None, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)


    # ======================================================================== #
    print("is done")
    sys.exit(0)

    for sweep in range(2, num_sweeps+1):
        rkix_obj = lambda block, t: (ix(sweep=sweep, rev=True, block=block, t=t, recon_level="object"))
        fkix_obj = lambda block, t: (ix(sweep=sweep, rev=False, block=block, t=t, recon_level="object"))

        for t in range(T):
            propose = Propose(
                # target=Reverse(partial(dec_coor2, recon_level='object'), enc_coor2, ix=rkix_obj("where", t)),

                # object
                target=Reverse(dec_coor2,                                      ## need LL of new samples of z_where
                               enc_coor2, ix=rkix_obj("where", t)),            ## <<<
                               # enc_coor2, ix=rkix_obj("frame", t)),            ## <<<

                proposal=Resample(Forward(enc_coor2,                           ## <<< need the conv_kernel as frame
                                          # propose, ix=fkix_obj("object", t)))) ## <<< need the conv_kernel as object
                                          propose, ix=fkix_obj("where", t)))) ## <<< need the conv_kernel as object

        propose = Propose(
            target=Reverse(dec_digit2, enc_digit2, ix=rkix_obj("what", T+1)),
            proposal=Resample(Forward(enc_digit2, propose, ix=fkix_obj("what", T+1))),
        )

    out = propose(dict(frames=frames, get_conv=get_conv, conv_kernel=None, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)

    # ========================================================================================#
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

    return out.loss, {f"loss{i}":v.detach().cpu().item() for i, v in enumerate(losses)}

def apg_comb(models, AT1, AT2, frames, K, result_flags, num_sweeps, resampler, mean_shape, device=None):
    loss = namedtuple('loss', ['phi', 'theta'])
    metrics = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_where' : [], 'E_recon' : [], 'density' : []}
    # log_w, q, metrics, propose_IS = oneshot(models, frames, mean_shape, metrics, result_flags)
    (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]
    T = frames.shape[2]
    conv_kernel = mean_shape
    loss0=loss(torch.zeros(1, **kw_autodevice(device)),torch.zeros(1, **kw_autodevice(device))),

    def loss_fn(cur, total_loss):
        ix = cur.proposal.ix
        jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)

        log_w = cur.log_weight.detach()
        w = F.softmax(log_w, 0)
        log_q = cur.proposal.log_prob if ix.block == "is" else \
            cur.proposal.log_prob - cur.proposal.program.trace.log_joint(**jkwargs)
        log_p = cur.target.log_prob
        # !!! need this metric as <density>

        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (-log_p)).sum(0).mean()

        return loss(phi=total_loss.phi + loss_phi, theta=total_loss.theta + loss_theta)

        # log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        # ppr(log_q, desc="log_q (hao)")
        # print('--------------------')
        # log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        # ppr(out.target.log_prob, desc='log_p (com)')
        # ppr(log_p,                 desc="log_p (hao)")
        # print('--------------------')
        # log_w = (log_p - log_q).detach()
        # ppr(out.log_weight, desc='log_w (com)')
        # ppr(log_w,          desc="log_w (hao)")
        # print('--------------------')
        # w = F.softmax(log_w, 0).detach()
        # print('onestep done')

    mkix_IS = lambda t: ix(sweep=1, rev=False, block='is', t=t, recon_level='frames')
    propose_IS = Noop()
    for t in range(T):
        propose_IS = Propose(
            ix=mkix_IS(t),
            target=dec_coor2,
            #       p_coor(c_1_t=0) ... p_coor(c_1_t=10)
            # ------------------------------------------------------
            #            q(z_1 | η_1, x) q_0(η_1|x)
            proposal=Forward(enc_coor2, propose_IS))

    propose_IS = Propose(ix=mkix_IS(T+1),
        target=dec_digit2,
        proposal=Resample(Forward(enc_digit2, propose_IS)))

    propose = propose_IS
    # ======================================================================== #
    for sweep in range(num_sweeps-1):
        rkix_obj = lambda block, t: (ix(sweep, True, block, t=t, recon_level="object"))
        fkix_obj = lambda block, t: (ix(sweep, False, block, t=t, recon_level="object"))

        for t in range(T):
            propose = Propose(
                # target=Reverse(partial(dec_coor2, recon_level='object'), enc_coor2, ix=rkix_obj("where", t)),
                target=Reverse(dec_coor2, enc_coor2, ix=rkix_obj("where", t)),
                proposal=Resample(Forward(enc_coor2, propose, ix=fkix_obj("where", t), _debug=True)))
            propose = propose

        propose = Propose(
            target=Reverse(dec_digit2, enc_digit2, ix=rkix_obj("what", t)),
            proposal=Resample(Forward(enc_digit2, propose, ix=fkix_obj("what", t))),
        )

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # def get_conv(ix, trace):
    #     if ix.block=='is':
    #         return mean_shape
    #     else:
    #         out = dec_digit2(q=trace, frames=frames, ix=ix)
    #         breakpoint();
    #         return out
    #
    # out = propose(dict(frames=frames, get_conv=get_conv, conv_kernel=None, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    # if result_flags['loss_required']:
    #     metrics['loss_phi'] = torch.cat(metrics['loss_phi'], 0)
    #     metrics['loss_theta'] = torch.cat(metrics['loss_theta'], 0)
    # if result_flags['ess_required']:
    #     metrics['ess'] = torch.cat(metrics['ess'], 0)
    # if result_flags['mode_required']:
    #     metrics['E_where'] = torch.cat(metrics['E_where'], 0)
    #     metrics['E_recon'] = torch.cat(metrics['E_recon'], 0)
    # if result_flags['density_required']:
    #     metrics['density'] = torch.cat(metrics['density'], 0)
    return metrics

def traverse(out, getter):
        current = out
        returns = []
        while True:
            if current.type == "Propose":
                returns.append(getter(current))
                if isinstance(propose.target, Program):
                    returns.reverse()
                    return returns
                current = current.proposal
            elif current.type in ["Resample", "Forward", "Reverse"]:
                current = current.program
            else:
                break

        returns.reverse()
        return returns

# def oneshot(models, frames, conv_kernel, metrics, result_flags):
#     (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
#     (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]
#     T = frames.shape[2]
#     S, B, K, DP, DP = conv_kernel.shape
#     q = probtorch.Trace()
#     p = probtorch.Trace()
#     print("========================================================")
#     def get_conv(ix, trace):
#         if ix.block=='is':
#             return conv_kernel # mean_shape
#         else:
#             out = dec_digit2(q=trace, frames=frames, ix=ix)
#             breakpoint();
#             return out
#
#     log_q = torch.zeros(S, B)
#     for t in range(T):
#         q = enc_coor(q, frames, t, conv_kernel, extend_dir='forward')
#         p = dec_coor(q, p, t)
#         log_q += q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
#         log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
#         log_w = (log_p - log_q)
#
#         if t <= IX:
#             print()
#             ppr(log_q, desc="log_q ({})\t{: .4f}\t".format(t, log_q.mean().item()))
#             ppr(log_p, desc="log_p ({})\t{: .4f}\t".format(t, log_p.mean().item()))
#             ppr(log_w, desc="log_w ({})\t{: .4f}\t".format(t, log_w.mean().item()))
#         # if t <= IX:
#         #     ppr(q, desc="hao prp trace")
#         #     ppr(p, desc="hao tar trace")
#         #     print()
#
#     print("--------------------------------------------------------")
#     # print("proposal trace:")
#     # ppr(q, m='vd')
#     # print("target trace:")
#     # ppr(p, m='vd')
#
#     # forward kernel shape
#     q = enc_digit(q, frames, timestep=T+1, extend_dir='forward')
#     # generative program shape
#     p = dec_digit(q, p, frames, timestep=T+1, recon_level='frames')
#
#     # print("POST: proposal trace:")
#     # ppr(q, m='vd')
#     # print("POST: target trace:")
#     # ppr(p, m='vd')
#
#
#     # (2/4 3:45pm)
#     mkix_IS = lambda t: ix(sweep=1, rev=False, block='is', t=t, recon_level='frames')
#     propose_IS = Noop()
#     for t in range(T):
#         propose_IS = Propose(
#             ix=mkix_IS(t),
#             target=dec_coor2,
#             #       p_coor(c_1_t=0) ... p_coor(c_1_t=10)
#             # ------------------------------------------------------
#             #            q(z_1 | η_1, x) q_0(η_1|x)
#             proposal=Forward(enc_coor2, propose_IS))
#
#     propose_IS = Propose(ix=mkix_IS(T+1),
#         target=dec_digit2,
#         proposal=Resample(Forward(enc_digit2, propose_IS)))
#
#     propose = propose_IS
#
#     out = propose_IS(dict(frames=frames, get_conv=get_conv, conv_kernel=None, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)
#
#
#     print('--------------------')
#     log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
#     ppr(log_q, desc="log_q (hao)")
#     print('--------------------')
#     log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
#     ppr(out.target.log_prob, desc='log_p (com)')
#     ppr(log_p,                 desc="log_p (hao)")
#     print('--------------------')
#     log_w = (log_p - log_q).detach()
#     ppr(out.log_weight, desc='log_w (com)')
#     ppr(log_w,          desc="log_w (hao)")
#     print('--------------------')
#     w = F.softmax(log_w, 0).detach()
#     print('onestep done')
#     sys.exit(0)
#
#     if result_flags['loss_required']:
#         loss_phi = (w * (- log_q)).sum(0).mean()
#         loss_theta = (w * (-log_p)).sum(0).mean()
#         metrics['loss_phi'].append(loss_phi.unsqueeze(0))
#         metrics['loss_theta'].append(loss_theta.unsqueeze(0))
#     if result_flags['ess_required']:
#         ess = (1. /(w**2).sum(0))
#         metrics['ess'].append(ess.unsqueeze(0))
#     if result_flags['mode_required']:
#         E_where = []
#         for t in range(T):
#             E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
#         E_where = torch.cat(z_where, 2)
#         metrics['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * K * 2
#         metrics['E_recon'].append(p['recon'].dist.probs.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * FP * FP
#     if result_flags['density_required']:
#         metrics['density'].append(log_p.detach().unsqueeze(0))
#     return log_w, q, metrics, propose_IS

def oneshot(models, frames, conv_kernel, metrics, result_flags):
    (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
    (enc_coor2, dec_coor2,enc_digit2, dec_digit2) = [models[k] for k in ["enc_coor2", "dec_coor2", "enc_digit2", "dec_digit2"]]
    T = frames.shape[2]
    S, B, K, DP, DP = conv_kernel.shape
    q = probtorch.Trace()
    p = probtorch.Trace()
    get_conv = _get_conv(conv_kernel, dec_digit2)
    print("========================================================")
    log_q = torch.zeros(S, B)
    debug.seed(0)
    for t in range(T):
        q = enc_coor(q, frames, t, conv_kernel, extend_dir='forward')
        p = dec_coor(q, p, t)
        log_q += q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
        log_w = (log_p - log_q)
    # forward kernel shape
    q = enc_digit(q, frames, timestep=T+1, extend_dir='forward')
    # generative program shape
    p = dec_digit(q, p, frames, timestep=T+1, recon_level='frames')

    log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = (log_p - log_q).detach()
    ppr(log_w, desc="IS complete  : log_w {:.4f}  <**> ".format(log_w.detach().mean().item()))
    pminus  = trace_utils.copysubtrace(p, set(p.keys()) - {'recon'})
    lpminus = pminus.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    ppr(p, desc="Target")
    ppr(q, desc="Proposal")
    print((lpminus - log_q).mean())
    print('--------------------------')
    print('     Combinators          ')
    print('--------------------------')

    w = F.softmax(log_w, 0).detach()

    # (2/4 3:45pm)
    mkix = lambda t: ix(sweep=1, rev=False, block='is', t=t, recon_level='frames')

    prop = Noop()
    for t in range(T):
        prop = Forward(kernel=enc_coor2, program=prop, ix=mkix(t))

    propose_IS = Propose(
        target=Forward(Memo(dec_digit2), dec_coor2, ix=mkix(T+1)),
        proposal=Forward(enc_digit2, prop, ix=mkix(T+1)))

    debug.seed(0)
    propose = propose_IS
    out = propose(dict(frames=frames, get_conv=get_conv, conv_kernel=None, z_where_T=[]), sample_dims=0, batch_dim=1, reparameterized=False)

    print('====================')
    ppr(out.proposal.trace, desc="Proposal(comb)", delim="\n, ")
    print("+++++++")
    ppr(out.target.trace, desc="Target  (comb)")
    print('====================')
    log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    pprm(log_q, name="log_q (hao)")
    pprm(out.proposal.log_prob,   name='log_q (com)')
    print('--------------------')
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    pprm(out.target.log_prob,   name='log_p (com)')
    pprm(log_p,                 name="log_p (hao)")
    print('--------------------')
    log_w = (log_p - log_q).detach()
    pprm(out.log_weight, name='log_w (com)')
    pprm(log_w,          name="log_w (hao)")
    print('--------------------')
    w = F.softmax(log_w, 0).detach()
    print('onestep done')
    sys.exit(0)

    ppr(out.log_weight, desc="IS combinator: log_w {:.4f}  <**> ".format(out.log_weight.detach().mean().item()))
    print('onestep done')
    breakpoint();


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
    print("========================================================")
    return log_w, q, metrics, None

def apg_where_t(models, frames, q, timestep, metrics, result_flags):
    T = frames.shape[2]
    (enc_coor, dec_coor, enc_digit, dec_digit) = [models[k] for k in ["enc_coor", "dec_coor", "enc_digit", "dec_digit"]]
    conv_kernel = dec_digit(q=q, p=None, frames=frames, timestep=timestep, recon_level='object')
    #ppr(conv_kernel, desc="conv_kernel")
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


