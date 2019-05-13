#!/usr/bin/env python3
###
import torch
import numpy
import probtorch
from probtorch.util import log_sum_exp

import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal, Uniform


def f_0(x):
    """ Target distribution: \propto N(3, 6) """
    return torch.exp(-(x-3)**2/6**2)
    # return torch.exp(Normal(3, 6).log_prob(x))

def f_n(x):
    """ Target distribution: \propto N(-4, 2) +  N(10, 2)"""
    return torch.exp(-(x+4)**2/2/2**2) + torch.exp(-(x-10)**2/2/2**2)

def f_j(x, beta):
    return f_0(x)**(1-beta) * f_n(x)**(beta)

def T(x, f, n_steps=1):
    n = x.shape[0]
    for i in range(n_steps):
        x_ = x + Normal(0, 1).sample((n,))
        a = f(x_) / f(x) 
        x += (x_ - x) * (Uniform(0, 1).sample((n,)) < a).float()
    return x


n_anneal = 50
n_samples = 100

p_0 = Normal(3, 6)
betas = torch.linspace(0, 1, n_anneal)

xs = torch.zeros((n_anneal, n_samples))
lws = torch.zeros((n_anneal, n_samples))
lw = torch.zeros(n_samples)

xs[0] = p_0.sample((n_samples,))
lws[0] = torch.log(f_0(xs[0])) - p_0.log_prob(xs[0])
for t in range(1, len(betas)):
    xs[t] = T(xs[t-1], lambda xs: f_j(xs, betas[t]), n_steps=5)
    lws[t] = lws[t-1] + torch.log(f_j(xs[t], betas[t])) - torch.log(f_j(xs[t], betas[t-1]))

ws = torch.exp(lws)
Zs = ws.mean(1)
Z_true = 2*numpy.sqrt(2*numpy.pi*2**2)

# Plot distributions
# x_range = torch.arange(-10, 20, 0.01)
# for beta in betas:
#     plt.plot(x_range.numpy(), f_j(x_range, beta).numpy())
# plt.scatter(xs[-1].numpy(), (ws[-1]/Zs[-1]).numpy())
# plt.show()

print('Z_true:', Z_true, ', ', 'Z_hat', Zs[-1].numpy())
###
