from torch.tensor import Tensor
from combinators.tensor.utils import autodevice
import os
import torch
import time
import numpy as np
from random import shuffle
from experiments.apgs_bshape.models import Enc_coor, Dec_coor, Enc_digit, Dec_digit
from experiments.apgs_bshape.hao import apg_objective
import os
import torch
import numpy as np
import argparse
from combinators import debug
from experiments.apgs_bshape.affine_transformer import Affine_Transformer
from experiments.apgs_bshape.dataset.loader import datapaths
from combinators.resampling.strategies import APGSResamplerOriginal
from combinators.utils import git_root

if debug.runtime() == 'jupyter':
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm
from tqdm.contrib import tenumerate

def train(
    objective,
    optimizer,
    models,
    AT,
    resampler,
    num_sweeps,
    data_paths,
    shape_mean,
    num_objects,
    num_epochs,
    sample_size,
    batch_size,
    device,
    model_version,
    checkpoint,
    checkpoint_filename
):
    """ training function of apg samplers """
    K = num_objects
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    is_smoketest = debug.is_smoketest()
    shape_mean = shape_mean.to(autodevice(device)).repeat(sample_size, batch_size, K, 1, 1)

    epochs = range(1) if is_smoketest else trange(num_epochs, desc='epochs', position=0)
    tdata_paths = [(0, data_paths[0])] if is_smoketest else tenumerate(data_paths, desc=f'group', position=1)

    for epoch in epochs:
        shuffle(data_paths)
        for group, data_path in tdata_paths:
            time_start = time.time()
            metrics = dict()
            data = torch.from_numpy(np.load(data_path)).float().to(autodevice(device))
            num_batches = int(data.shape[0] / batch_size)
            seq_indices = torch.randperm(data.shape[0])
            with trange(3 if is_smoketest else num_batches, desc=f'batch', position=2) as batches:
                for b in batches:
                    optimizer.zero_grad()
                    frames = data[seq_indices[b*batch_size : (b+1)*batch_size]].repeat(sample_size, 1, 1, 1, 1)
                    trace = objective(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean)
                    loss_phi = trace['loss_phi'].sum()
                    loss_theta = trace['loss_theta'].sum()
                    loss_phi.backward(retain_graph=True)
                    loss_theta.backward()

                    optimizer.step()
                    if 'loss_phi' in metrics:
                        metrics['loss_phi'] += trace['loss_phi'][-1].item()
                    else:
                        metrics['loss_phi'] = trace['loss_phi'][-1].item()
                    if 'loss_theta' in metrics:
                        metrics['loss_theta'] += trace['loss_theta'][-1].item()
                    else:
                        metrics['loss_theta'] = trace['loss_theta'][-1].item()
                    if 'ess' in metrics:
                        metrics['ess'] += trace['ess'][-1].mean().item()
                    else:
                        metrics['ess'] = trace['ess'][-1].mean().item()
                    if 'density' in metrics:
                        metrics['density'] += trace['density'][-1].mean().item()
                    else:
                        metrics['density'] = trace['density'][-1].mean().item()

                    batches.set_postfix_str(",  ".join(['{}={:.4f}'.format(k, v[-1].mean().item()/num_batches) for k, v in trace.items() if isinstance(v, Tensor)]))
                if checkpoint:
                    save_models(models, checkpoint_filename)

                metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
                if not os.path.exists('results/'):
                    os.makedirs('results/')
                log_file = open('results/log-' + model_version + '.txt', 'a+' if epoch==0 else 'w+')
                if not is_smoketest:
                    tdata_paths.set_postfix_str(metrics_print)

                time_end = time.time()
                print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
                log_file.close()


# def train(optimizer, models, AT, resampler, num_sweeps, data_paths, shape_mean, K, num_epochs, sample_size, batch_size, CUDA, device, model_version):
#     """
#     training function of apg samplers
#     """
#     result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
#     shape_mean = shape_mean.repeat(sample_size, batch_size, K, 1, 1)
#     for epoch in range(num_epochs):
#         shuffle(data_paths)
#         for group, data_path in enumerate(data_paths):
#             time_start = time.time()
#             metrics = dict()
#             data = torch.from_numpy(np.load(data_path)).float()
#             num_batches = int(data.shape[0] / batch_size)
#             seq_indices = torch.randperm(data.shape[0])
#             for b in range(num_batches):
#                 optimizer.zero_grad()
#                 frames = data[seq_indices[b*batch_size : (b+1)*batch_size]].repeat(sample_size, 1, 1, 1, 1)
#                 if CUDA:
#                     with torch.cuda.device(device):
#                         frames = frames.cuda()
#                         shape_mean = shape_mean.cuda()
#                 trace = apg_objective(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean)
#                 loss_phi = trace['loss_phi'].sum()
#                 loss_theta = trace['loss_theta'].sum()
#                 loss_phi.backward(retain_graph=True)
#                 loss_theta.backward()
#                 optimizer.step()
#                 if 'loss_phi' in metrics:
#                     metrics['loss_phi'] += trace['loss_phi'][-1].item()
#                 else:
#                     metrics['loss_phi'] = trace['loss_phi'][-1].item()
#                 if 'loss_theta' in metrics:
#                     metrics['loss_theta'] += trace['loss_theta'][-1].item()
#                 else:
#                     metrics['loss_theta'] = trace['loss_theta'][-1].item()
#                 if 'ess' in metrics:
#                     metrics['ess'] += trace['ess'][-1].mean().item()
#                 else:
#                     metrics['ess'] = trace['ess'][-1].mean().item()
#                 if 'density' in metrics:
#                     metrics['density'] += trace['density'][-1].mean().item()
#                 else:
#                     metrics['density'] = trace['density'][-1].mean().item()
#             save_models(models, model_version)
#             metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
#             if not os.path.exists('results/'):
#                 os.makedirs('results/')
#             if epoch == 0 and group == 0:
#                 log_file = open('results/log-' + model_version + '.txt', 'w+')
#             else:
#                 log_file = open('results/log-' + model_version + '.txt', 'a+')
#             time_end = time.time()
#             print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
#             log_file.close()
#             print("Epoch=%d, Group=%d completed in (%ds),  " % (epoch, group, time_end - time_start))

def init_models(AT, frame_pixels, digit_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, CUDA, device, load_version, lr):
    enc_coor = Enc_coor(num_pixels=(frame_pixels-digit_pixels+1)**2, num_hidden=num_hidden_coor, z_where_dim=z_where_dim, AT=AT)
    dec_coor = Dec_coor(z_where_dim=z_where_dim, CUDA=CUDA, device=device)
    enc_digit = Enc_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT)
    dec_digit = Dec_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT, CUDA=CUDA, device=device)
    if CUDA:
        with torch.cuda.device(device):
            enc_coor.cuda()
            enc_digit.cuda()
            dec_digit.cuda()

    if load_version is not None:
        weights = torch.load("weights/cp-%s" % load_version)
        enc_coor.load_state_dict(weights['enc-coor'])
        enc_digit.load_state_dict(weights['enc-digit'])
        dec_digit.load_state_dict(weights['dec-digit'])
    if lr is not None:
        optimizer =  torch.optim.Adam(list(enc_coor.parameters())+
                                        list(enc_digit.parameters())+
                                        list(dec_digit.parameters()),
                                        lr=lr,
                                        betas=(0.9, 0.99))
#         optimizer =  torch.optim.SGD(list(enc_coor.parameters())+
#                                         list(enc_digit.parameters())+
#                                         list(dec_digit.parameters()),
#                                         lr=lr)
        return (enc_coor, dec_coor, enc_digit, dec_digit), optimizer
#     else:
#         for p in enc_coor.parameters():
#             p.requires_grad = False
#         for p in enc_digit.parameters():
#             p.requires_grad = False
#         for p in dec_digit.parameters():
#             p.requires_grad = False
    return (enc_coor, dec_coor, enc_digit, dec_digit)

def save_models(models, save_version):
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    checkpoint = {
        'enc-coor' : enc_coor.state_dict(),
        'enc-digit' : enc_digit.state_dict(),
        'dec-digit' : dec_digit.state_dict()
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, 'weights/cp-%s' % save_version)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--data_dir', default='../../data/bshape/')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--num_sweeps', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--num_objects', default=2, type=int)
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--frame_pixels', default=40, type=int)
    parser.add_argument('--shape_pixels', default=10, type=int)
    parser.add_argument('--num_hidden_digit', default=200, type=int)
    parser.add_argument('--num_hidden_coor', default=200, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    args = parser.parse_args()

    sample_size = int(args.budget / args.num_sweeps)
    debug.seed(0)
    device = torch.device('cpu') # 'cuda:%d' % args.device)

    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-bshape-num_objects=%s-num_samples=%s' % (args.num_objects, sample_size)
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-bshape-num_objects=%s-num_sweeps=%s-num_samples=%s' % (args.num_objects, args.num_sweeps, sample_size)
    else:
        raise ValueError

    data_paths = datapaths(data_dir=args.data_dir, subfolder='')
    shape_mean = torch.from_numpy(np.load('dataset/shape_mean.npy')).float()

    AT = Affine_Transformer(args.frame_pixels, args.shape_pixels).to(autodevice(device))
    resampler = APGSResamplerOriginal(sample_size).to(autodevice(device))
    # models = init_models(AT, args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, load=False, device=device)
    # optimizer = adam(models, lr=args.lr)

    CUDA=torch.cuda.is_available()
    models, optimizer = init_models(AT, args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, CUDA, device, load_version=None, lr=args.lr)
    print('Start training for bshape tracking task..')
    print('version=' + model_version)
    train(
        objective=apg_objective,
        optimizer=optimizer,
        models=models,
        AT=AT,
        resampler=resampler,
        num_sweeps=args.num_sweeps,
        data_paths=data_paths,
        shape_mean=shape_mean,
        num_objects=args.num_objects,
        num_epochs=args.num_epochs,
        sample_size=sample_size,
        batch_size=args.batch_size,
        device=device,
        model_version=model_version,
        checkpoint=False,
        checkpoint_filename=model_version)
    # train(
    #     objective=apg_objective_declarative,
    #     optimizer=optimizer,
    #     models=models,
    #     AT=AT,
    #     resampler=resampler,
    #     num_sweeps=num_sweeps,
    #     data_paths=data_paths,
    #     shape_mean=shape_mean,
    #     num_objects=args.num_objects,
    #     num_epochs=args.num_epochs,
    #     sample_size=sample_size,
    #     batch_size=args.batch_size,
    #     device=device,
    #     model_version=model_version,
    #     checkpoint=True,
    #     checkpoint_filename=model_version
    # )

# /usr/bin/env python3
# import torch
# import time
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim, Tensor
# import math
# from typing import NoReturn, Tuple
# import os
# import torch
# import time
# import numpy as np
# from random import shuffle
# import operator
# from itertools import accumulate
# from functools import partial
# from collections import namedtuple
# from torch.distributions.one_hot_categorical import OneHotCategorical as cat
#
# from combinators.inference import _dispatch
# from combinators import Forward, Reverse, Propose, Condition, Resample, RequiresGrad, Program
# from combinators.metrics import effective_sample_size, log_Z_hat
# from combinators.tensor.utils import autodevice, kw_autodevice
# from combinators.stochastic import Trace
# import combinators.trace.utils as trace_utils
# import combinators.tensor.utils as tensor_utils
# import combinators.debug as debug
# from combinators.utils import ppr, curry
# import sys
# from combinators.trace.utils import RequiresGrad, copysubtrace, copytrace, mapvalues, disteq
#
# from combinators.utils import adam, save_models
# from combinators.tensor.utils import autodevice
#
# if debug.runtime() == 'jupyter':
#     from tqdm.notebook import trange, tqdm
# else:
#     from tqdm import trange, tqdm
# from tqdm.contrib import tenumerate
#
# import experiments.apgs_bshape.hao as hao
# from experiments.apgs_bshape.dataset.loader import datapaths
# from experiments.apgs_bshape.models import Enc_coor, Dec_coor, Enc_digit, Dec_digit, init_models, ix
#
# def apg_objective_declarative(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean):
#     # (enc_rws_eta, enc_apg_z, enc_apg_eta, generative, x, sample_size, num_sweeps, compare=True)
#     compare = True
#     if compare:
#         # frames = frames[:,:,0:1,:,:]
#         # frames = frames[:,:,:,:,:]
#         pass
#     T = frames.shape[2]
#     if compare:
#         debug.seed(1)
#         from combinators.resampling.strategies import APGSResamplerOriginal
#         sweeps, metrics = hao.apg_objective(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean)
#         print(sweeps[1]['trace']['iloss_phi'])
#         print(sweeps[1]['trace']['iloss_theta'])
#         debug.seed(1)
#     print('hao done')
#     sys.exit(0)
#
#     # xs ::  T * S * B * K * D
#     isix = ix(sweep=1,rev=False,block='is',t=None)
#
#     T = frames.shape[2]
#     S, B, K, DP, DP = digit.shape
#     # for t in range(T):
#
#     # is_step_ = Propose(
#     #     loss_fn=loss_fn,
#     #     target=generative, ix=isix,
#     #     #                   p(x,η_1,z_1)
#     #     #        ---------------------------------------
#     #     #            q(z_1 | η_1 x) q_rws(η_1 | x)
#     #     proposal=Forward(enc_apg_z, enc_rws_eta, ix=isix))
#
#     # def loss_fn(cur, total_loss):
#     #     ix = cur.proposal.ix
#     #     jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
#     #
#     #     log_w = cur.log_weight.detach()
#     #     w = F.softmax(log_w, 0)
#     #     log_q = cur.proposal.log_prob if ix.block == "is" else \
#     #         cur.proposal.log_prob - cur.proposal.program.trace.log_joint(**jkwargs)
#     #
#     #     batch_loss = (w * (- log_q)).sum(0)
#     #     if ix.block == 'z':
#     #         batch_loss = batch_loss.sum(-1) # account for one-hot encoding
#     #     return batch_loss.mean() + total_loss
#
#
#     # propose = Resample(is_step)
#     #
#     # for sweep in range(2, num_sweeps+1):
#     #     mkix = lambda block: (ix(sweep, True, block), ix(sweep, False, block))
#     #     rix, fix = mkix('eta')
#     #
#     #     propose_eta = Propose(loss_fn=loss_fn,
#     #         target=Reverse(propose, enc_apg_eta, ix=rix),
#     #         #                p(x η_2 z_1) q(η_1 | z_1 x)
#     #         #           --------------------------------------
#     #         #                 q(η_2 | z_1 x) p(η_1,x,z_1)
#     #         proposal=Resample(Forward(enc_apg_eta, propose, ix=fix), strategy=APGResampler))
#     #
#     #     rix, fix = mkix('z')
#     #     propose_z = Propose(loss_fn=loss_fn,
#     #         target=Reverse(generative, enc_apg_z, ix=rix),
#     #         #             p(x η_2 z_2) q( z_1 | η_2 x)
#     #         #        ----------------------------------------
#     #         #            q(z_2 | η_2 x) p(η_2 x z_1)
#     #         proposal=Resample(Forward(enc_apg_z, propose_eta)), ix=fix)
#     #
#     #     propose = propose_z
#     out = propose(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=compare)
#
#     def traverse(out, getter):
#         current = out
#         returns = []
#         while True:
#             if current.type == "Propose":
#                 returns.append(getter(current))
#                 if isinstance(propose.target, Program):
#                     returns.reverse()
#                     return returns
#                 current = current.proposal
#             elif current.type in ["Resample", "Forward", "Reverse"]:
#                 current = current.program
#             else:
#                 break
#
#         returns.reverse()
#         return returns
#
#     losses = [out.loss] # traverse(out, lambda p: p.loss.squeeze())
#     if compare:
#         losses_hao = list(accumulate(sweeps[1]['metrics']['iloss'], operator.add))
#         assert len(losses) == len(losses_hao)
#         mismatch_losses = list(filter(lambda xs: not torch.equal(*xs), zip(losses, losses_hao)))
#         if len(mismatch_losses) > 0:
#             for l, r in mismatch_losses:
#                 print(l, "!=", r)
#             assert False
#
#
#     return out.loss, {f"loss{i}":v.detach().cpu().item() for i, v in enumerate(losses)}
#
#
#
#
#
# # def apg():
# #     propose = Resample(is_step)
#
# #     for sweep in range(2, num_sweeps+1):
# #         mkix = lambda block: (ix(sweep, True, block), ix(sweep, False, block))
# #         rix, fix = mkix('eta')
# #         pr1 =
# #         for t in range(timesteps):
# #             pr1 = Propose(loss_fn=loss_fn,
# #                 target=Reverse(propose, enc_apg_eta, ix=(rix, t)),
# #                 #                p(x η_2 z_1) q(η_1 | z_1 x)
# #                 #           --------------------------------------
# #                 #                 q(η_2 | z_1 x) p(η_1,x,z_1)
# #                 proposal=Resample(Forward(enc_apg_eta, pr1, ix=fix), strategy=APGResampler))
#
# #         rix, fix = mkix('z')
# #         propose_z = Propose(loss_fn=loss_fn,
# #             target=Reverse(generative, enc_apg_z, ix=rix),
# #             #             p(x η_2 z_2) q( z_1 | η_2 x)
# #             #        ----------------------------------------
# #             #            q(z_2 | η_2 x) p(η_2 x z_1)
# #             proposal=Resample(Forward(enc_apg_z, propose_eta)), ix=fix)
#
# #         propose = propose_z
#     out = propose(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=compare)
