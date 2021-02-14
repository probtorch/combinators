import os
import time
import torch
import argparse
import numpy as np
from random import shuffle
from experiments.apgs_bshape.models import Enc_coor, Enc_digit, Decoder
from experiments.apgs_bshape.affine_transformer import Affine_Transformer

from combinators.tensor.utils import autodevice
from combinators.utils import adam, git_root, save_models, load_models
from combinators import debug
# from combinators.resampling.strategies import APGSResamplerOriginal


# if debug.runtime() == 'jupyter':
#     from tqdm.notebook import trange, tqdm
# else:
#     from tqdm import trange, tqdm
# from tqdm.contrib import tenumerate

def train(optimizer, models, AT, resampler, num_sweeps, data_paths, shape_mean, K, num_epochs, sample_size, batch_size, CUDA, device, model_version):
    """
    training function of apg samplers
    """
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    shape_mean = shape_mean.repeat(sample_size, batch_size, K, 1, 1)
    for epoch in range(num_epochs):
        shuffle(data_paths)
        for group, data_path in enumerate(data_paths):
            time_start = time.time()
            metrics = dict()
            data = torch.from_numpy(np.load(data_path)).float()
            num_batches = int(data.shape[0] / batch_size)
            seq_indices = torch.randperm(data.shape[0])
            for b in range(num_batches):
                optimizer.zero_grad()
                frames = data[seq_indices[b*batch_size : (b+1)*batch_size]].repeat(sample_size, 1, 1, 1, 1)
                if CUDA:
                    with torch.cuda.device(device):
                        frames = frames.cuda()
                        shape_mean = shape_mean.cuda()
                trace = apg_objective(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean)
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
            save_models(models, model_version)
            metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
            if not os.path.exists('results/'):
                os.makedirs('results/')
            if epoch == 0 and group == 0:
                log_file = open('results/log-' + model_version + '.txt', 'w+')
            else:
                log_file = open('results/log-' + model_version + '.txt', 'a+')
            time_end = time.time()
            print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
            log_file.close()


            
def init_models(frame_pixels, shape_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, device):
    models = dict()
    AT = Affine_Transformer(frame_pixels, shape_pixels).cuda().to(device)

    models['enc-coor'] = Enc_coor(num_pixels=(frame_pixels-shape_pixels+1)**2, 
                                    num_hidden=num_hidden_coor, 
                                    z_where_dim=z_where_dim, 
                                    AT=AT).cuda().to(device)
    
    models['enc-digit'] = Enc_digit(num_pixels=shape_pixels**2, 
                                      num_hidden=num_hidden_digit, 
                                      z_what_dim=z_what_dim, 
                                      AT=AT).cuda().to(device)

    models['dec'] = Decoder(num_pixels=shape_pixels**2, 
                              num_hidden=num_hidden_digit, 
                              z_where_dim=z_where_dim, 
                              z_what_dim=z_what_dim, 
                              AT=AT).cuda().to(device)
    return models

def save_models(models, save_version):
    checkpoint = dict()
    for k,v in models.items():
        checkpoint[k] = v.state_dict()
    if not os.path.exists('./weights/'):
        os.makedirs('./weights/')
    torch.save(checkpoint, 'weights/cp-%s' % save_version)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--data_dir', default='./dataset/video/')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--budget', default=120, type=int)
    parser.add_argument('--num_sweeps', default=6, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--frame_pixels', default=96, type=int)
    parser.add_argument('--shape_pixels', default=28, type=int)
    parser.add_argument('--num_hidden_digit', default=400, type=int)
    parser.add_argument('--num_hidden_coor', default=400, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    args = parser.parse_args()

    sample_size = int(args.budget / args.num_sweeps)
    device = torch.device('cuda:%d' % args.device)

    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-bshape-num_objects=%s-num_samples=%s' % (args.num_objects, sample_size)
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-bshape-num_objects=%s-num_sweeps=%s-num_samples=%s' % (args.num_objects, args.num_sweeps, sample_size)
    else:
        raise ValueError

    data_paths = datapaths(data_dir=args.data_dir, subfolder='')
    mean_shape = torch.load('./dataset/mean_shape.pt').cuda().to(device)

    models = init_models(args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, device)
    
    optimizer = torch.optim.Adam([v.parameters() for k,v in models.items()],lr=args.lr,betas=(0.9, 0.99))
        
    print('Start training for bshape tracking task..')
    print('version=' + model_version)
#     train(
#         # objective=apg_comb,
#         objective=apg_objective,
#         optimizer=optimizer,
#         models=models,
#         AT1=AT1,
#         AT2=AT2,
#         resampler=resampler,
#         num_sweeps=args.num_sweeps,
#         data_paths=data_paths,
#         shape_mean=shape_mean,
#         num_objects=args.num_objects,
#         num_epochs=args.num_epochs,
#         sample_size=sample_size,
#         batch_size=args.batch_size,
#         device=device,
#         model_version=model_version,
#         checkpoint=False,
#         checkpoint_filename=model_version)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
