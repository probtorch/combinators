#!/usr/bin/env python3
try:
    # The default backend; one of GTK GTKAgg GTKCairo GTK3Agg GTK3Cairo
    # CocoaAgg MacOSX Qt4Agg Qt5Agg TkAgg WX WXAgg Agg Cairo GDK PS PDF SVG
    import matplotlib
    matplotlib.use('TKAgg')
except:
    pass

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import colors
from scipy.interpolate import interpn
from matplotlib import cm


def scatter(xs, lws=None, c='C0', ax=None, show=False):
    xs = xs.squeeze().detach().cpu().numpy()
    assert len(xs.shape) == 2
    inplace = ax is not None
    cm_endpoints = [(i, (*colors.to_rgb(c), i)) for i in [0.0, 1.0]]
    lin_alpha = colors.LinearSegmentedColormap.from_list('incr_alpha', cm_endpoints)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(*xs.T, c=None if lws is None else lws.softmax(dim=0), cmap=lin_alpha)

    if show:
        plt.show()
    return fig if fig is not None else ax

def scatter_along(samples):
    fig = plt.figure(figsize=(5*len(samples), 5))
    gspec = gridspec.GridSpec(ncols=len(samples), nrows=1, figure=fig)

    for i, xs in enumerate(samples):
        ax = fig.add_subplot(gspec[0, i])
        scatter(xs, ax=ax)
    return fig

# NVI PLOTTING ============================================================


def plot_sample_hist(ax, x, y, sort=True, bins=20, range=None, weight_cm=False, **kwargs):
    ax.tick_params(bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    mz, x_e, y_e = numpy.histogram2d(x.numpy(), y.numpy(), bins=bins, density=True, range=range)
    if weight_cm:
        z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), mz, numpy.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)
        z[numpy.where(numpy.isnan(z))] = 0.0  # To be sure to plot all data
        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
        cmap = cm.get_cmap('viridis')
        ax.set_facecolor(cmap(0.))
        ax.scatter(x, y, c=z, cmap=cmap, **kwargs)
        return mz
    else:
        ax.imshow(mz)

# Plot densities
def plot_density_contour(ax, dim, xls, log_density_fn, yls=None, normalize=False, **kwargs):
    if dim == 1:
        plot_density_1d(ax, xls, log_density_fn, **kwargs)
    elif dim == 2:
        if yls is None:
            yls = xls
        mx, my, mxy = _mkmesh(xls, yls)
        lmz = log_density_fn(mxy).detach()
        mz = lmz.exp()
        if normalize:
            mz, Z = _normalize(xls, yls, mz)
        ax.contour(mx, my, mz, **kwargs)

def plot_density_image(ax, xls, log_density_fn, yls=None, normalize=False, **kwargs):
    if yls is None:
        yls = xls
    mx, my, mxy = _mkmesh(xls, yls)
    lmz = log_density_fn(mxy).detach()
    mz = lmz.exp()
    if normalize:
        mz, Z = _normalize(xls, yls, mz)
    ax.imshow(mz, **kwargs)

def plot_density_surface(ax, dim, xls, log_density_fn, yls=None, normalize=False, **kwargs):
    if dim == 1:
        plot_density_1d(ax, xls, log_density_fn, **kwargs)
    elif dim == 2:
        if yls is None:
            yls = xls
        mx, my, mxy = _mkmesh(xls, yls)
        lmz = log_density_fn(mxy).detach()
        mz = lmz.exp()
        if normalize:
            mz, Z = _normalize(xls, yls, mz)
        ax.plot_wireframe(mx, my, mz, **kwargs)
    else:
        raise NotImplementedError

def plot_density_1d(ax, xls, log_density_fn, **kwargs):
    mx = torch.arange(*xls)[:, None]
    lp = log_density_fn(mx).detach()
    ax.plot(mx, lp.exp(), **kwargs)

# Plot kernels

def plot_kernel_displacement(ax, xls, proposal, target, kernel,
                             angles='xy', scale_units='xy', pivot='tail', **kwargs):
    mx, my, mxy = _mkmesh(xls)
    means = kernel.get_params({'z_in': mxy})['loc']
    mean_displacement = means - mxy
    ax.quiver(mx, my,
              mean_displacement.detach()[:, :, 0],
              mean_displacement.detach()[:, :, 1],
              numpy.hypot(mean_displacement.detach()[:, :, 0],
                          mean_displacement.detach()[:, :, 1]),
              angles=angles,
              scale_units=scale_units,
              pivot=pivot,
              **kwargs)

def plot_kernel_sample_scatter(ax, zs_in, proposal, target, kernel, *args, rev_kernel=None,
                               weight_alphas=False, resample=False, **kwargs):
    zs_out = kernel.get_dist({'z_in': zs_in}).sample()
    lws = None
    if resample:
        if rev_kernel is None:
            raise ValueError("resampling only possible if reverse kernel is provided")
        lws = _balanced_seq_log_weight(proposal, target, kernel, rev_kernel, zs_in, zs_out)
        zs_out, _ = resample_systematic(zs_out, lws)

    plot_sample_scatter(ax, zs_out, *args, lws=lws, weight_alphas=weight_alphas, **kwargs)
    return zs_out

def plot_kernel_sample_hist(ax, zs_in, proposal, target, kernel, *args, rev_kernel=None,
                            bins=150, resample=False, **kwargs):
    zs_out = kernel.get_dist({'z_in': zs_in}).sample()
    if resample:
        if rev_kernel is None:
            raise ValueError("resampling only possible if reverse kernel is provided")
        lw = _balanced_seq_log_weight(proposal, target, kernel, rev_kernel, zs_in, zs_out)
        zs_out, _ = resample_systematic(zs_out, lw)

    plot_sample_hist(ax, *zs_out.squeeze().t(), bins=bins, **kwargs)
    return zs_out

# Helpers
def _normalize(xls, yls, grid):
    dx, dy = xls[2], yls[2]
    Z = grid.sum()*dx*dy
    return grid/Z, Z

def _mkmesh(xls, yls=None, fn=None,):
    if yls is None:
        yls = xls
    xrange = torch.arange(*xls)
    yrange = torch.arange(*yls)
    mx, my = torch.meshgrid(xrange, yrange)
    mxy = torch.stack((mx, my), dim=-1)
    if fn is None:
        return mx, my, mxy
    return mx, my, fn(mxy).detach()

def _levels_gaussian(gaussian, nstd=3):
    std_vec = torch.eye(gaussian.loc.ndim)[0] * gaussian.covariance_matrix[0, 0].sqrt()
    levels = [gaussian.log_prob(gaussian.loc + k*std_vec).exp() for k in range(nstd, 0, -1)]
    return levels

        # def get_gaussian_levels(gaussian, n_levels):
        #     levels = [gaussian.log_prob(gaussian.loc + k * gaussian.covariance_matrix[0]).exp()
        #               for k in [2, 1]
        #               ]
        #     return levels

def _balanced_seq_log_weight(proposal, target, fwd_kernel, rev_kernel, z1, z2):
    target_lp = target.get_log_density()(z2) + rev_kernel.get_log_density({'z_in': z2})(z1)
    proposal_lp = proposal.get_log_density()(z1) + fwd_kernel.get_log_density({'z_in': z1})(z2)
    return target_lp - proposal_lp

def _get_rejection_samples(xls, log_density_fn):
    mx, my, mxy = _mkmesh(xls)
    zs_uniform = torch.stack((mx.flatten(), my.flatten()), dim=-1)
    zs, _ = resample_systematic(zs_uniform[:, None], log_density_fn(zs_uniform[:, None]))
    return zs.squeeze()

def plot_mean_std(ax, means, stds, *args, nstd=2, alpha_mean=0.8, alpha_std=0.3, label=None, **kwargs):
    ax.plot(means, *args, label=label, alpha=alpha_mean, **kwargs)
    xrange = torch.arange(means.shape[-1])
    ax.fill_between(xrange, means-nstd*stds, means+nstd*stds, alpha=alpha_std, **kwargs)
