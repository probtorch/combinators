import torch
import time
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import shuffle
from apgs.resampler import Resampler
from apgs.bshape.objectives import apg_objective, bpg_objective, hmc_objective
from apgs.bshape.hmc_sampler import HMC

    
def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    
def density_all_instances(models, AT, data_paths, sample_size, K, z_where_dim, z_what_dim, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device, batch_size=10):
    densities = dict()
    ess = dict()
#     shuffle(data_paths)
    data = torch.from_numpy(np.load(data_paths[0])).float()
    num_batches = 5 # int(data.shape[0] / batch_size)
    mnist_mean = torch.from_numpy(np.load('shape_mean.npy')).float()
    mnist_mean = mnist_mean.repeat(sample_size, batch_size, K, 1, 1)
    for b in range(num_batches):
        print('batch=%d/%d' % (b+1, num_batches))
        time_start = time.time()
        x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1, 1)
        if CUDA:
            x = x.cuda().to(device)            
            mnist_mean = mnist_mean.cuda().to(device)
        S, B, T, FP, _ = x.shape
        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(models, AT, S, B, T, K, z_where_dim, z_what_dim, num_sweeps, lf_step_size, lf_step_size, lf, CUDA, device)
            trace_hmc = hmc_objective(models, AT, x, result_flags, hmc_sampler, mnist_mean) 
            if 'HMC-RWS(L=%d, LF=%d)' % (S, lf) in densities:
                densities['HMC-RWS(L=%d, K=%d, LF=%d)' % (S, num_sweeps, lf)].append(trace_hmc['density'][-1].mean(-1).mean(-1).cpu().numpy())
#                 ess['HMC-RWS(L=%d, K=%d, LF=%d)' % (S, K, lf)].append(trace_hmc['ess'][-1].mean(-1).cpu().numpy())
            else:
                densities['HMC-RWS(L=%d, K=%d, LF=%d)' % (S, num_sweeps, lf)] = [trace_hmc['density'][-1].mean(-1).mean(-1).cpu().numpy()]
#                 ess['HMC-RWS(L=%d, K=%d, LF=%d)' % (S, K, lf)] = [trace_hmc['ess'][-1].mean(-1).cpu().numpy()]
                
#         x_bpg = x.repeat(bpg_factor, 1, 1, 1, 1)
#         mnist_mean_bpg = mnist_mean.repeat(bpg_factor, 1, 1, 1, 1)
#         trace_bpg = bpg_objective(models, AT, x_bpg, result_flags, num_sweeps, resampler_bpg, mnist_mean_bpg)
#         if 'BPG(L=%d)' % (S*bpg_factor) in densities:
#             densities['BPG(L=%d)' % (S*bpg_factor)].append(trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
#         else:
#             densities['BPG(L=%d)' % (S*bpg_factor)] = [trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
#             ess['BPG(L=%d)' % (S*bpg_factor)] = [trace_apg['ess'][-1].mean(-1).cpu().numpy()]
#         trace_apg = apg_objective(models, AT, x, K, result_flags, num_sweeps, resampler, mnist_mean)
#         if 'APG(L=%d, K=%d)' % (S, num_sweeps) in densities:
#             densities['APG(L=%d, K=%d)' % (S, num_sweeps)].append(trace_apg['density'][-1].mean(-1).mean(-1).cpu().numpy())
#             ess['APG(L=%d, K=%d)' % (S, num_sweeps)].append(trace_apg['ess'][-1].mean(-1).cpu().numpy())
#         else:
#             densities['APG(L=%d, K=%d)' % (S, num_sweeps)] = [trace_apg['density'][-1].mean(-1).mean(-1).cpu().numpy()]
#             ess['APG(L=%d, K=%d)' % (S, num_sweeps)] = [trace_apg['ess'][-1].mean(-1).cpu().numpy()]
        time_end = time.time()
        print('%d / %d completed in (%ds)' % (b+1, num_batches, time_end-time_start))
    for key in densities.keys():
        densities[key] = np.array(densities[key]).mean()
#         ess[key] =np.array(ess[key]).mean()
        print('method=%s, log joint=%.2f' % (key, densities[key]))


def viz_samples(frames, rs, ws, num_sweeps, num_objects, object_size, fs=1, title_fontsize=12, lw=2, colors=['#AA3377', '#EE7733', '#009988', '#0077BB', '#BBBBBB', '#EE3377', '#DDCC77'], save=False):
    B, T, FP, _ = frames.shape
#     recons_first = rs[0]
#     z_wheres_first = ws[0].clone()
#     z_wheres_first[:,:,:,1] =  z_wheres_first[:,:,:,1] * (-1)
#     c_pixels_first = z_wheres_first
#     c_pixels_first = (c_pixels_first + 1.0) * (FP - object_size) / 2. # B * T * K * D
    
    recons_last = rs[0]
    z_wheres_last = ws[0].clone()
    z_wheres_last[:,:,:,1] =  z_wheres_last[:,:,:,1] * (-1)
    c_pixels_last = z_wheres_last
    c_pixels_last = (c_pixels_last + 1.0) * (FP - object_size) / 2. # B * T * K * D
    for b in range(B):
        num_cols = T
        num_rows =  2
#         c_pixel_first, recon_first = c_pixels_first[b].numpy(), recons_first[b].numpy()
        c_pixel_last, recon_last = c_pixels_last[b].numpy(), recons_last[b].numpy()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for c in range(num_cols):
#             ax_infer = fig.add_subplot(gs[0, c])
#             ax_infer.imshow(frames[b, c].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
#             ax_infer.set_xticks([])
#             ax_infer.set_yticks([])
#             for k in range(K):
#                 rect_k = patches.Rectangle((c_pixel_first[c, k, :]), object_size, object_size, linewidth=lw, edgecolor=colors[k],facecolor='none')
#                 ax_infer.add_patch(rect_k)
#             ax_recon = fig.add_subplot(gs[1, c])
#             ax_recon.imshow(recon_first[c], cmap='gray', vmin=0.0, vmax=1.0)
#             ax_recon.set_xticks([])
#             ax_recon.set_yticks([])
            
            ax_infer = fig.add_subplot(gs[0, c])
            ax_infer.imshow(frames[b, c].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            ax_infer.set_xticks([])
            ax_infer.set_yticks([])
            for k in range(num_objects):
                rect_k = patches.Rectangle((c_pixel_last[c, k, :]), object_size, object_size, linewidth=lw, edgecolor=colors[k],facecolor='none')
                ax_infer.add_patch(rect_k)
            ax_recon = fig.add_subplot(gs[1, c])
            ax_recon.imshow(recon_last[c], cmap='gray', vmin=0.0, vmax=1.0)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
    if save:
        plt.savefig('combinators_apg_samples.svg', dpi=300)

def plot_convergence(densities, fs=6, fs_title=14, lw=3, opacity=0.1, colors = ['#0077BB', '#009988', '#EE7733', '#AA3377', '#555555', '#999933']):
    fig = plt.figure(figsize=(fs*2.5,fs)) 
    ax = fig.add_subplot(111)
    i = 0
    for key, value in densities.items():
        mean, std = value.mean(0), value.std(0)
        ax.plot(mean, linewidth=lw, c=colors[i], label=key)
        ax.fill_between(np.arange(len(mean)), mean-std, mean+std, color=colors[i], alpha=opacity)
        i += 1
    ax.legend(fontsize=10, loc='lower right')
    ax.tick_params(labelsize=15)
    ax.set_xlabel('Sweeps', fontsize=25)
    ax.set_ylabel(r'$\log \: p_\theta(x, z)$', fontsize=25)
    ax.grid(alpha=0.4)
    
def density_convergence(models, AT, data, sample_size, K, num_runs, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device):
    DENSITIES = dict()
    shape_mean = torch.from_numpy(np.load('shape_mean.npy')).float()
    shape_mean = mnist_mean.repeat(sample_size, batch_size, K, 1, 1)
    x = data[torch.randperm(data.shape[0])[0]].repeat(sample_size, 1, 1, 1)
    if CUDA:
        x = x.cuda().to(device)
    for i in range(num_runs):
        densities = dict()
        set_seed(i)    
        S, B, T, D = x.shape
        
        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(models, AT, S, B, T, K, D, num_sweeps, lf_step_size, lf, CUDA, device)
#             print('Running RWS-HMC with %dx leapfrog steps..' % lp)
            _, _, trace_hmc = hmc_objective(models, x, result_flags, hmc_sampler) 
            densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)] = trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[None, :] 
#         print('Running Standard Gibbs..')
        trace_gibbs = gibbs_objective(models, x, result_flags, num_sweeps)
        densities['GIBBS(L=%d)' % S] = trace_gibbs['density'].mean(-1).mean(-1).cpu().numpy()[None, :] 
#         print('Running Bootstrapped Population Gibbs..')
        x_bpg = x.repeat(bpg_factor, 1, 1, 1)
        trace_bpg = bpg_objective(models, x_bpg, result_flags, num_sweeps, resampler_bpg)
        densities['BPG(L=%d)' % (S*bpg_factor)] = trace_bpg['density'] .mean(-1).mean(-1).cpu().numpy()[None, :]
#         print('Running Amortized Population Gibbs..')
        block = 'decomposed'
        trace_apg = apg_objective(models, x, result_flags, num_sweeps, block, resampler)
        densities['APG(L=%d)' % S] = trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[None, :]
        for key, value in densities.items():
            if key in DENSITIES:
                DENSITIES[key].append(value)
            else:
                DENSITIES[key] = [value]
        print('Run=%d / %d completed..' % (i+1, num_runs))
    for key in DENSITIES.keys():
        DENSITIES[key] = np.concatenate(DENSITIES[key], 0)
    return DENSITIES 


def budget_analysis(models, blocks, num_sweeps, sample_sizes, data, K, CUDA, device, batch_size=100):
    """
    compute the ess and log joint under same budget
    """
    result_flags = {'loss_required' : False, 'ess_required' : True, 'mode_required' : False, 'density_required': True}

    ess = []
    density = []
    num_batches = int((data.shape[0] / batch_size))
    metrics = {'block' : [], 'num_sweeps' : [], 'sample_sizes' : [], 'ess' : [], 'density' : []}
    for block in blocks:
        for i in range(len(num_sweeps)):
            metrics['block'].append(block)
            time_start = time.time()
            num_sweep = int(num_sweeps[i])
            sample_size = int(sample_sizes[i])
            metrics['num_sweeps'].append(num_sweep)
            metrics['sample_sizes'].append(sample_size)
            resampler = Resampler(strategy='systematic',
                                  sample_size=sample_size,
                                  CUDA=CUDA,
                                  device=device)
            ess, density = 0.0, 0.0
            for b in range(num_batches):
                x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)
                if CUDA:
                    x = x.cuda().to(device)
                trace = apg_objective(models, x, result_flags, num_sweeps=num_sweep, block=block, resampler=resampler)
                ess += trace['ess'][-1].mean().item()
                density += trace['density'][-1].mean().item()
            metrics['ess'].append(ess / num_batches / sample_size)
            metrics['density'].append(density / num_batches)
            time_end = time.time()
            print('block=%s, num_sweep=%d, sample_size=%d completed in %ds' % (block, num_sweep, sample_size, time_end-time_start))
    return pd.DataFrame.from_dict(metrics)
            
            
            
def plot_budget_analyais_results(df, fs=8, fs_title=14, lw=3, fontsize=20, colors=['#AA3377', '#009988', '#EE7733', '#0077BB', '#BBBBBB', '#EE3377', '#DDCC77']):
    """
    plot the results of budget analysis
    """
    df_decomposed = df.loc[df['block'] == 'decomposed']
    df_joint = df.loc[df['block'] == 'joint']
    ticklabels = []
    num_sweeps = df_decomposed['num_sweeps'].to_numpy()
    sample_sizes = df_decomposed['sample_sizes'].to_numpy()
    for i in range(len(num_sweeps)):
        ticklabels.append('K=%d\nL=%d' % (num_sweeps[i], sample_sizes[i]))
    fig = plt.figure(figsize=(fs*2.5, fs))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(num_sweeps, df_decomposed['density'].to_numpy(), 'o-', c=colors[0], linewidth=lw, label=r'$\{\mu, \tau\}, \{c\}$')
    ax1.plot(num_sweeps, df_joint['density'].to_numpy(), 'o-', c=colors[1],  linewidth=lw,label=r'$\{\mu, \tau, c\}$')
    ax1.set_xticks(num_sweeps)
    ax1.set_xticklabels(ticklabels)
    ax1.tick_params(labelsize=fontsize)
    ax1.grid(alpha=0.4)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(num_sweeps, df_decomposed['ess'].to_numpy(), 'o-', c=colors[0],  linewidth=lw,label=r'$\{\mu, \tau\}, \{c\}$')
    ax2.plot(num_sweeps, df_joint['ess'].to_numpy(), 'o-', c=colors[1],  linewidth=lw,label=r'$\{\mu, \tau, c\}$')
    ax2.set_xticks(num_sweeps)
    ax2.set_xticklabels(ticklabels)
    ax2.tick_params(labelsize=fontsize)
    ax2.grid(alpha=0.4)
    ax2.legend(fontsize=fontsize)
    ax1.legend(fontsize=fontsize)
    ax1.set_ylabel(r'$\log \: p_\theta(x, \: z)$', fontsize=35)
    ax2.set_ylabel('ESS / L', fontsize=35) 