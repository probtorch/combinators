#!/usr/bin/env python3
from combinators.utils import adam
from combinators.tensor.utils import autodevice
import os
import torch
import time
import numpy as np
from random import shuffle
from experiments.apgs_bshape.models import Enc_coor, Dec_coor, Enc_digit, Dec_digit, init_models
from experiments.apgs_bshape.objectives import apg_objective

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
            log_file = open('results/log-' + model_version + '.txt', 'a+' if epoch==0 else 'w+')
            time_end = time.time()
            print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
            log_file.close()
            print("Epoch=%d, Group=%d completed in (%ds),  " % (epoch, group, time_end - time_start))




def apg():
    propose = Resample(is_step)

    for sweep in range(2, num_sweeps+1):
        mkix = lambda block: (ix(sweep, True, block), ix(sweep, False, block))
        rix, fix = mkix('eta')
        pr1 =
        for t in range(timesteps):
            pr1 = Propose(loss_fn=loss_fn,
                target=Reverse(propose, enc_apg_eta, ix=(rix, t)),
                #                p(x η_2 z_1) q(η_1 | z_1 x)
                #           --------------------------------------
                #                 q(η_2 | z_1 x) p(η_1,x,z_1)
                proposal=Resample(Forward(enc_apg_eta, pr1, ix=fix), strategy=APGResampler))

        rix, fix = mkix('z')
        propose_z = Propose(loss_fn=loss_fn,
            target=Reverse(generative, enc_apg_z, ix=rix),
            #             p(x η_2 z_2) q( z_1 | η_2 x)
            #        ----------------------------------------
            #            q(z_2 | η_2 x) p(η_2 x z_1)
            proposal=Resample(Forward(enc_apg_z, propose_eta)), ix=fix)

        propose = propose_z
    out = propose(x=x, prior_ng=generative.prior_ng, sample_dims=0, batch_dim=1, reparameterized=False, _debug=compare)


if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    import argparse
    from combinators import debug
    from experiments.apgs_bshape.affine_transformer import Affine_Transformer
    from combinators.resampling.strategies import APGSResamplerOriginal
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--data_dir', default='../../data/bshape/')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--num_sweeps', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
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
    device = torch.device('cuda:%d' % args.device)

    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-bshape-num_objects=%s-num_samples=%s' % (args.num_objects, sample_size)
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-bshape-num_objects=%s-num_sweeps=%s-num_samples=%s' % (args.num_objects, args.num_sweeps, sample_size)
    else:
        raise ValueError

    data_paths = []
    for file in os.listdir(args.data_dir + '%dobjects/train/' % args.num_objects):
        data_paths.append(os.path.join(args.data_dir, '%dobjects/train' % args.num_objects, file))

    shape_mean = torch.from_numpy(np.load('shape_mean.npy')).float()

    AT = Affine_Transformer(args.frame_pixels, args.shape_pixels).to(autodevice(device))
    resampler = APGSResamplerOriginal(sample_size).to(autodevice(device))

    models = init_models(args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, load=False, device=device)
    optimizer = adam(models, lr=args.lr)

    print('Start training for bshape tracking task..')
    print('version=' + model_version)
    train(optimizer, models, AT, resampler, args.num_sweeps, data_paths, shape_mean, args.num_objects, args.num_epochs, sample_size, args.batch_size, CUDA, device, model_version)
