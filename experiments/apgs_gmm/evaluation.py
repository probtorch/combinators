import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from combinators.resampling.strategies import APGResampler
from combinators.objectives import apg_objective, bpg_objective, gibbs_objective, hmc_objective
from experiments.apgs_gmm.hmc_sampler import HMC


def density_all_instances(models, data, sample_size, K, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device, batch_size=100):
    densities = dict()
    num_batches = int(data.shape[0] / batch_size)
    for b in range(num_batches):
        time_start = time.time()
        x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)
        if CUDA:
            x = x.cuda().to(device)
        S, B, N, D = x.shape

        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(S, B, N, K, D, num_sweeps, lf_step_size, lf, CUDA, device)
            _, _, trace_hmc = hmc_objective(models, x, result_flags, hmc_sampler)
            if 'HMC-RWS(L=%d, LF=%d)' % (S, lf) in densities:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)].append(trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1])
            else:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)] = [trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        trace_gibbs = gibbs_objective(models, x, result_flags, num_sweeps)
        if 'GIBBS(L=%d)' % S in densities:
            densities['GIBBS(L=%d)' % S].append(trace_gibbs['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['GIBBS(L=%d)' % S] = [trace_gibbs['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        x_bpg = x.repeat(bpg_factor, 1, 1, 1)
        trace_bpg = bpg_objective(models, x_bpg, result_flags, num_sweeps, resampler_bpg)
        if 'BPG(L=%d)' % (S*bpg_factor) in densities:
            densities['BPG(L=%d)' % (S*bpg_factor)].append(trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['BPG(L=%d)' % (S*bpg_factor)] = [trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        block = 'decomposed'
        trace_apg = apg_objective(models, x, result_flags, num_sweeps, block, resampler)
        if 'APG(L=%d)' % S in densities:
            densities['APG(L=%d)' % S].append(trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['APG(L=%d)' % S] = [trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        time_end = time.time()
        print('%d / %d completed in (%ds)' % (b+1, num_batches, time_end-time_start))
    for key in densities.keys():
        densities[key] = np.array(densities[key]).mean()
        print('method=%s, log joint=%.2f' % (key, densities[key]))

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

def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def density_convergence(models, data, sample_size, K, num_runs, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device):
    DENSITIES = dict()
    x = data[torch.randperm(data.shape[0])[0]].repeat(sample_size, 1, 1, 1)
    if CUDA:
        x = x.cuda().to(device)
    for i in range(num_runs):
        densities = dict()
        set_seed(i)
        S, B, N, D = x.shape

        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(S, B, N, K, D, num_sweeps, lf_step_size, lf, CUDA, device)
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

#################################################
#          visualize  samples                   #
#                                               #
#################################################
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

def viz_gmm(ax, ob, K, marker_size, opacity, bound, colors, latents=None):
    if latents == None:
        ax.scatter(ob[:, 0], ob[:, 1], c='k', s=marker_size, zorder=3)
    else:
        (tau, mu, state) = latents
        sigma2 = 1. / tau
        assignments = state.argmax(-1)
        for k in range(K):
            cov_k = np.diag(sigma2[k])
            ob_k = ob[np.where(assignments == k)]
            ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], s=marker_size, zorder=3)
            plot_cov_ellipse(cov=cov_k, pos=mu[k], nstd=2, color=colors[k], ax=ax, alpha=opacity, zorder=3)
    ax.set_ylim([-bound, bound])
    ax.set_xlim([-bound, bound])
    ax.set_xticks([])
    ax.set_yticks([])

def viz_samples(data, trace, num_sweeps, K, viz_interval=3, figure_size=3, title_fontsize=20, marker_size=1.0, opacity=0.3, bound=20, colors=['#AA3377','#0077BB', '#EE7733', '#009988', '#BBBBBB', '#EE3377', '#DDCC77'], save_name=None):
    """
    visualize the samples along the sweeps
    """
    E_tau, E_mu, E_z = trace['E_tau'].cpu(), trace['E_mu'].cpu(), trace['E_z'].cpu()
    num_rows = len(data)
    num_cols = 2 + int((num_sweeps-1) / viz_interval)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size * num_cols, figure_size * num_rows))
    for row_ind in range(num_rows):
        ax = fig.add_subplot(gs[row_ind, 0])
        viz_gmm(ax, data[row_ind], K, marker_size, opacity, bound, colors, latents=None) ## visualize raw dataset in the 1st column
        if row_ind == 0:
            ax.set_title('Data', fontsize=title_fontsize)
#         col_ind = 1
        for col_ind in range(num_cols-1):
            sweep = col_ind * viz_interval
            ax = fig.add_subplot(gs[row_ind, col_ind+1])
            viz_gmm(ax, data[row_ind], K, marker_size, opacity, bound, colors, latents=(E_tau[sweep, row_ind], E_mu[sweep, row_ind], E_z[sweep, row_ind]))
            if row_ind == 0:
                if sweep == 0:
                    ax.set_title('RWS', fontsize=title_fontsize)
                else:
                    ax.set_title('sweep %d' % sweep, fontsize=title_fontsize)
    if save_name is not None:
        plt.savefig(save_name + '.svg', dpi=300)
