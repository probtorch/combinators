#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.distributions as dist
from probtorch import Trace, Factor
from torch import Tensor
from typing import Callable, Any, Tuple, Optional
from collections import ChainMap

class TwoGaussians(Dataset):
    def __init__(self):
        dist0 = dist.Normal(0,1)
        dist1 = dist.Normal(0,1)
        fst = dist0.sample(sample_shape=torch.size(1,50))
        snd = dist0.sample(sample_shape=torch.size(1,50))
        self.alldat = torch.concat([fst , snd])
        self.order = torch.randperm(100)

    def __get__(self, i):
        return self.alldat[self.order[i]]


class Normal(dist.normal.Normal):
    def __init__(self, loc, scale, validate_args=None, sample_shape=1):
        super().__int__(*args, **kwargs)
        self.sample_shape = sample_shape

    def __invert__(self):
        return self.sample(self.sample_shape)


g = nn.Sequential(nn.Linear(1, 5), nn.ReLU(), nn.Linear(5, 10))
e = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

def p(g, x):
    tr = Trace()
    z = tr.normal(torch.zeros([1]), torch.ones([1]), name="z")
    a = tr.normal(             z  , torch.ones([1]), name="a")
    tr.normal(g(z), torch.ones([1]), value=x, name="x")
    return x, z

@model(args=[], kwargs=[])
def p(tr, g, x):
    z = tr.normal(torch.zeros([1]), torch.ones([1]), name="z")
    a = tr.normal(             z  , torch.ones([1]), name="a")
    tr.normal(g(z), torch.ones([1]), value=x, name="x")
    return x, z

@model(args=[], kwargs=[])
def p(tr, g):
    z = tr.normal(torch.zeros([1]), torch.ones([1]), name="z")
    a = tr.normal(             z  , torch.ones([1]), name="a")
    x = tr.normal(           g(z) , torch.ones([1]), name="x")
    return x, z
random, _ = p()
p2 = condition(p, {'x': 0.5})  # >>> model with x initialized to 0.5, identical to observed RVs!!!
0.5, _    = p2()

#### DANGER ZONE
q = query(p, 'a') # q() => value of 'a' in trace

[1,2,3]
 _
[1,4,3]

[1,4,5]





def q(e, f, x):
    tr = Trace()
    m, s = e(x)
    b = tr.normal(m,             s  , name="b")
    u = tr.normal(b, torch.ones([1]), name="u")
    Factor(f(u, x))
    return u, x

def f(k, u, x):
    tr = Trace()
    m, s = k(u, x)
    z = tr.normal(m, s, name="z")
    return z

def r(k, x, z):
    tr = Trace()
    m, s = k(z, x)
    u = tr.normal(m, s, name="u")
    return u

if __name__ == '__main__':
    xs = TwoGaussians()
    breakpoint()
