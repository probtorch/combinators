#!/usr/bin/env python3

import torch
from torch.distributions import Uniform, MultivariateNormal, Normal
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
    def _forward(self, *args, data={}):
        width = torch.ones(*self.batch_shape) * data['image'].shape[1]
        height = torch.ones(*self.batch_shape) * data['image'].shape[0]
        return self.sample(Uniform, torch.zeros(*self.batch_shape, 2),
                           torch.stack((width, height), dim=-1), name=self.name)

class ImageProposalMH(model.Primitive):
    def _forward(self, *args, data={}):
        width = torch.ones(*self.batch_shape) * data['image'].shape[1]
        height = torch.ones(*self.batch_shape) * data['image'].shape[0]
        wh = torch.stack((width, height), dim=-1)
        return self.sample(MultivariateNormal, loc=wh/2.,
                           covariance_matrix=torch.eye(2) * wh/4.,
                           name=self.name)

class AnnealingProposal(inference.Inference):
    def __init__(self, target, annealing_steps):
        super(AnnealingProposal, self).__init__(target)
        self._annealing_steps = annealing_steps

    def cond(self, qs):
        return AnnealingProposal(self.target.cond(qs), self._annealing_steps)

    def walk(self, f):
        return f(AnnealingProposal(self.target.walk(f), self._annealing_steps))

    def forward(self, *args, t=0, **kwargs):
        beta = torch.linspace(1e-6, 1, self._annealing_steps)[t]

        zs, xi, log_weight = self.target(*args, **kwargs)
        xi[self.target.name].factor(log_weight * (1 - beta),
                                    name='AnnealingProposal')
        log_weight += xi[self.target.name]['AnnealingProposal'].log_prob

        return (zs, beta, kwargs.pop('data', {})), xi, log_weight

class ProbtorchLogoDensity(model.Primitive):
    def _forward(self, coords, beta, data={}):
        grid_density = (1.0 - data['image']/255)
        density = bilinearInterpolation(grid_density, coords[:, 0],
                                        coords[:, 1])
        self.factor((density + 1e-6).log() * beta, name='image')

        return coords

if __name__ == '__main__':
    from scipy.misc import imread
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt

    img_ary = imread('probtorch-logo-bw.png', mode='L')
    img_ary = gaussian_filter(img_ary, sigma=0.1)
    grid_density = torch.FloatTensor(1 - img_ary/255)

    n = 1000
    x = torch.rand(n) * grid_density.shape[1]
    y = torch.rand(n) * grid_density.shape[0]
    fxy = bilinearInterpolation(grid_density, x, y).numpy()
    # img_ary = numpy.array([[1.,2.],[3.,4.]]) # Dummy image for testing

    plt.matshow(img_ary)
    plt.scatter(x, y, c='r')
    plt.scatter(x[(fxy > 0.5).nonzero()], y[(fxy > 0.5).nonzero()], c='b')
    plt.show()
