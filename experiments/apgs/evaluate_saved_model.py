#!/usr/bin/env python3

import os
import torch
import numpy as np
from combinators.utils import load_models

from experiments.apgs.gibbs import gibbs_sweeps
from experiments.apgs.models import init_models

data_dir = './dataset/'
timesteps = 10
frame_pixels = 96
shape_pixels = 28
num_objects = 3

device = 'cuda:1'
num_epochs = 1000
lr = 2e-4
batch_size = 2
budget = 120
num_sweeps = 5

num_hidden_digit = 400
num_hidden_coor = 400
z_where_dim = 2
z_what_dim = 10
use_markov_blanket = False

device = torch.device(device)
sample_size = budget // (num_sweeps + 1)
assert sample_size > 0, 'non-positive sample size =%d' % sample_size
mean_shape = torch.load(data_dir + 'mean_shape.pt').to(device)
data_paths = []
for file in os.listdir(data_dir+'/video/'):
    if file.endswith('.pt') and \
    'timesteps=%d-objects=%d' % (timesteps, num_objects) in file:
        data_paths.append(os.path.join(data_dir+'/video/', file))
if len(data_paths) == 0:
    raise ValueError('Empty data path list.')

frames = torch.load(data_paths[0])[3:3+batch_size,:timesteps]

frames_expand = frames.to(device).repeat(sample_size, 1, 1, 1, 1)
print(frames.shape)

model_version = 'apg-timesteps=%d-objects=%d-sweeps=%d-samples=%d%s' % (timesteps, num_objects, num_sweeps, sample_size, "-with_opt" if use_markov_blanket else "")
models = init_models(mean_shape=mean_shape, frame_pixels=frame_pixels, shape_pixels=shape_pixels, num_hidden_digit=num_hidden_digit, num_hidden_coor=num_hidden_coor, z_where_dim=z_where_dim, z_what_dim=z_what_dim, num_objects=num_objects, device=device, use_markov_blanket=use_markov_blanket)

print(f"loading cp-{model_version}")
weight_file = f"cp-{model_version}"

load_models(models, weight_file)

apg = gibbs_sweeps(models, num_sweeps, timesteps)

out, frames = apg({"frames": frames_expand}, sample_dims=0, batch_dim=1, reparameterized=True), frames


def get_samples(out, sweeps, T):
    assert not use_markov_blanket, "can't use 'recon' with optimization"
    recon_vals = out.trace['recon'].dist.probs
    z_where_vals = torch.zeros(*frames_expand.shape[:-2],  num_objects, z_where_dim)
    for t in range(T):
        z_where_vals[:, :, t, :, :] = out.trace['z_where_%d_%d'%(t,sweeps)].value.detach().cpu()

    return (recon_vals.detach().cpu(), z_where_vals.detach().cpu())

rs, ws = get_samples(out, num_sweeps, timesteps)

# Visualize samples

from experiments.apgs_bshape.utils import viz_samples
viz_samples(frames, rs, ws, num_sweeps, num_objects, shape_pixels, fs=1, save=True)

# FIXME: figure out these next
# # Compute joint across all methods
# from apgs.bshape.evaluation import density_all_instances
# from random import shuffle
#
# sample_size, num_sweeps = 20, 5
# lf_step_size, lf_num_steps, bpg_factor = 5e-5, [100], 1
# density_all_instances(models, AT, data_paths, sample_size, num_objects, z_where_dim, z_what_dim, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device)
#
# from apgs.bshape.evaluation import budget_analysis, plot_budget_analyais_results
# data = torch.from_numpy(np.load(data_dir + '%dobjects/test/ob-1.npy' % num_objects)).float()
# budget = 1000
# num_sweeps = np.array([1, 5, 10 , 20, 25])
# sample_sizes = 1000 / num_sweeps
# blocks = ['decomposed', 'joint']
# df = budget_analysis(models, blocks, num_sweeps, sample_sizes, data, num_objects, CUDA, device)
# plot_budget_analyais_results(df)


