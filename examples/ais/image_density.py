#!/usr/bin/env python3

import torch
from torch.distributions import Normal
import numpy

import combinators.inference as inference
from combinators.sampler import Sampler
import combinators.model as model
import combinators.utils as utils

dtype = torch.FloatTensor
dtype_long = torch.LongTensor

def bilinearInterpolation(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return Ia*wa + Ib*wb + Ic*wc + Id*wd

class ImageProposal(model.Primitive):
    def _forward(self, *args, **kwargs):
        return self.sample(Normal, torch.zeros(*self.batch_shape, 2),
                           torch.ones(*self.batch_shape, 2), name=self.name)

class AnnealingProposal(inference.Inference):
    def __init__(self, target, annealing_steps):
        super(AnnealingProposal, self).__init__(target)
        self._annealing_steps = annealing_steps

    def cond(self, qs):
        return AnnealingProposal(self.target.cond(qs), self._annealing_steps)

    def walk(self, f):
        return f(AnnealingProposal(self.target.walk(f), self._annealing_steps))

    def forward(self, *args, t=0, **kwargs):
        beta = torch.linspace(0, 1, self._annealing_steps)[t]

        zs, xi, log_weight = self.target(*args, **kwargs)
        xi[self.target.name].factor(log_weight * (1 - beta),
                                    name='AnnealingProposal')
        log_weight += xi[self.target.name]['AnnealingProposal'].log_prob

        return (zs, beta, kwargs.pop('data', {})), xi, log_weight

class ProbtorchLogoDensity(model.Primitive):
    def _forward(self, coords, beta, data={}):
        grid_density = -(data['image'] - 255).t()
        grid_density = grid_density.expand(*self.batch_shape,
                                           *grid_density.shape)

        # Determine grid cell boundaries
        floor = torch.floor(coords).to(dtype=torch.long)
        ceil = torch.ceil(coords).to(dtype=torch.long)

        # Bilinear interpolation for density values at coordinates
        density_floor = (ceil[:, 0].to(dtype=torch.float32) - coords[:, 0]) * grid_density[floor[:, 0], floor[:, 1]] +\
                        (coords[:, 0] - floor[0].to(dtype=torch.float32)) * grid_density[ceil[:, 0], floor[:, 1]]
        density_ceil = (ceil[:, 0].to(dtype=torch.float32) - coords[:, 0]) * grid_density[floor[:, 0], ceil[:, 1]] +\
                       (coords[:, 0] - floor[:, 0].to(dtype=torch.float32)) * grid_density[ceil[:, 0], ceil[:, 1]]
        density = (ceil[:, 1] - coords[:, 1]) * density_floor +\
                  (coords[:, 1] - floor[:, 1]) * density_ceil
        self.factor(density.log() * beta, name='image')

        return coords

if __name__ == '__main__':
    from scipy.misc import imread
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt

    img_ary = imread('probtorch-logo-bw.png', mode='L')
    img_ary = gaussian_filter(img_ary, sigma=0.1)
    grid_density = torch.FloatTensor(-(img_ary - 255).T)

    n = 100
    x = torch.rand(n) * grid_density.shape[0]
    y = torch.rand(n) * grid_density.shape[1]
    fxy = bilinearInterpolation(grid_density, x, y).numpy()
    # img_ary = numpy.array([[1.,2.],[3.,4.]]) # Dummy image for testing

    plt.matshow(img_ary)
    plt.scatter(x, y, c='r')
    plt.scatter(x[(fxy > 0.5).nonzero()], y[(fxy > 0.5).nonzero()], c='b')
    plt.show()
