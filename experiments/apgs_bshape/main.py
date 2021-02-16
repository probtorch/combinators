import os
import time
import argparse
import torch
import numpy as np
from random import shuffle
from torch import optim, Tensor, nn

from combinators import debug
from combinators import git_root, save_models, load_models
from combinators import ImproperRandomVariable, RandomVariable, Trace, Program, Extend, Compose, Propose, Condition, Resample
from combinators import trace_utils, adam, ppr, autodevice, kw_autodevice, nvo_rkl, nvo_avo, effective_sample_size, log_Z_hat

from experiments.apgs_bshape.gibbs import gibbs_sweeps
from experiments.apgs_bshape.models import init_models

# from combinators.resampling.strategies import APGSResamplerOriginal
# if debug.runtime() == 'jupyter':
#     from tqdm.notebook import trange, tqdm
# else:
#     from tqdm import trange, tqdm
# from tqdm.contrib import tenumerate

def train(optimizer, models, num_sweeps, data_paths, num_objects, num_epochs, sample_size, batch_size, device, model_version):
    """
    training function of apg samplers
    """
#     result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
#     shape_mean = shape_mean.repeat(sample_size, batch_size, K, 1, 1)
    for epoch in range(num_epochs):
        shuffle(data_paths)
        for group, data_path in enumerate(data_paths):
            time_start = time.time()
            metrics = dict()
            data = torch.load(data_path)
            num_batches = int(data.shape[0] / batch_size)
            seq_indices = torch.randperm(data.shape[0])
            for b in range(num_batches):
                optimizer.zero_grad()
                frames = data[seq_indices[b*batch_size : (b+1)*batch_size]].repeat(sample_size, 1, 1, 1, 1).to(device)                
                apg = gibbs_sweeps(models, num_sweeps, T)
                apg(c={"frames": frames}, sample_dims=0, batch_dim=1, reparameterized=False)
    

                loss.backward()
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
    return models

def run_apg():
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
    mean_shape = torch.load('./dataset/mean_shape.pt').to(device)
    models = init_models(args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, args.num_objects, mean_shape, device)
    optimizer = optim.Adam([v.parameters() for k,v in models.items()],lr=args.lr,betas=(0.9, 0.99))
    data_paths = []
    for file in os.listdir(args.data_dir):
        if file.endswith('.pt'):
            data_paths.append(os.path.join(args.data_dir, file))
    print('Start training for bshape tracking task..')
    print('version=' + model_version)
    
    
    
    train(
        objective=gibbs_sweeps,
        optimizer=optimizer,
        models=models,
        AT1=AT1,
        AT2=AT2,
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

def test_gibbs_sweep(sweeps, T, sample_size, data_dir='./dataset/video/', mean_shape_path='./dataset/mean_shape.pt', **kwargs):
    device = torch.device(kwargs['device'])
    mean_shape = torch.load(mean_shape_path).to(device)
    models = init_models(mean_shape=mean_shape, **kwargs)

    data_paths = []
    for file in os.listdir(data_dir):
        if file.endswith('.pt'):
            data_paths.append(os.path.join(data_dir, file))
    frames_path = data_paths[0]
    frames = torch.load(frames_path)[:,:T]
    frames_expand = frames.to(device).repeat(sample_size, 1, 1, 1, 1)

    apg = gibbs_sweeps(models, sweeps, T)
    c = {"frames": frames_expand}
    return apg(c, sample_dims=0, batch_dim=1, reparameterized=False), frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--data_dir', default='./dataset/video/')
    parser.add_argument('--frame_pixels', default=96, type=int)
    parser.add_argument('--shape_pixels', default=28, type=int)
    parser.add_argument('--num_hidden_digit', default=400, type=int)
    parser.add_argument('--num_hidden_coor', default=400, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--mean_shape_path', default='./dataset/mean_shape.pt', type=str)
    
    args = parser.parse_args()
    test_gibbs_sweep(sweeps=5, 
                     T=10, 
                     data_dir=args.data_dir,
                     frame_pixels=args.frame_pixels, 
                     shape_pixels=args.shape_pixels, 
                     num_hidden_digit=args.num_hidden_digit, 
                     num_hidden_coor=args.num_hidden_coor, 
                     z_where_dim=args.z_where_dim, 
                     z_what_dim=args.z_what_dim, 
                     num_objects=args.num_objects, 
                     sample_size=args.sample_size,
                     mean_shape_path=args.mean_shape_path,
                     device=args.device)

