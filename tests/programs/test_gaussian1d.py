#!/usr/bin/env python3

import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn as nn
from torch.nn.parameter import Parameter
from combinators import Program, Kernel, Forward, Reverse, Propose, Trace
from combinators.stochastic import RandomVariable, Provenance
from typeguard import typechecked
from tqdm import trange
from pytest import mark
import combinators.trace.utils as trace_utils

from typing import Iterable, Optional

class FFNormalParams(nn.Module):
    def __init__(self, shapes, std=None): # None makes this learnable
        assert std is None or type(std) == float
        super().__init__()
        if std is None:
            raise NotImplementedError("still don't have learnable std yet")
        import ipdb; ipdb.set_trace();

        # self.joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out))
        # self.mu = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out))
        # self.sigma = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out), nn.ReLU())
        self.dim_out = dim_out

    def forward(self, x):
        # y = self.joint(x)
        mu = self.mu(x)
        # sigma = self.sigma(y)
        return mu, torch.ones([self.dim_out])*0.5 # sigma


class _RGaussian1d(nn.Module):
    def __init__(self, name:str, loc:float, scale:Optional[float], num_samples=1):
        super().__init__()
        self.name = name
        expand_samples = lambda ten: ten.expand(num_samples, *ten.size())

        self.register_parameter('loc', Parameter(expand_samples(torch.tensor([loc], dtype=torch.float)), requires_grad=True))

        learn_scale = scale is not None
        _scale = scale if learn_scale else 1
        scale_tensor = expand_samples(torch.pow(torch.tensor([_scale], dtype=torch.float), 2))
        if learn_scale:
            self.register_parameter('scale', Parameter(scale_tensor, requires_grad=True))
        else:
            self.register_buffer('scale', scale_tensor)

    def forward(self, trace_name, cond_trace=None):
        trace = Trace()
        out = trace.normal(
            loc=self.loc,
            scale=self.scale,
            value=None if cond_trace is None else cond_trace[trace_name].value,
            name=trace_name)
        return out, trace

    def __repr__(self):
        return "[{}]{}".format(self.name, super().__repr__())

class _RGaussian1d(nn.Module):
    def __init__(self, name:str, loc:float, scale:Optional[float], num_samples=1):
        super().__init__()
        self.name = name
        self.loc = loc
        expand_samples = lambda ten: ten.expand(num_samples, *ten.size())
        self.mk_loc = lambda: expand_samples(torch.tensor([loc], dtype=torch.float, requires_grad=True))
        # self.register_parameter('loc', Parameter(self.mk_loc(), requires_grad=True))

        learn_scale = scale is not None
        _scale = scale if learn_scale else 1
        self.mk_scale = lambda: expand_samples(torch.pow(torch.tensor([_scale], dtype=torch.float), 2))
        # if learn_scale:
        #     self.register_parameter('scale', Parameter(self.mk_scale(), requires_grad=True))
        # else:
        #     self.register_buffer('scale', self.mk_scale())

    def forward(self, trace_name, cond_trace=None):
        trace = Trace()
        out = trace.normal(
            loc=self.mk_loc(),
            scale=self.mk_scale(),
            value=None if cond_trace is None else cond_trace[trace_name].value,
            name=trace_name)
        return out, trace

    def __repr__(self):
        return "[{}]{}".format(self.name, super().__repr__())

def mkAdam(models:Iterable[nn.Module], **kwargs):
    return torch.optim.Adam([dict(params=x.parameters()) for x in models], **kwargs)

# def test_rgaussian1d_has_params():
#     mean = _RGaussian1d(loc=0, scale=None, name='g', num_samples=10)
#     assert len(list(mean.parameters())) == 1
#     full = _RGaussian1d(loc=0, scale=1, name='g', num_samples=10)
#     assert len(list(full.parameters())) == 2


def test_rgaussian1d_can_learn():
    mean = _RGaussian1d(loc=0, scale=None, name='g', num_samples=10)
    param = mean.mk_loc()
    #assert torch.equal(mean.loc, torch.zeros_like(param))
    #optimizer = torch.optim.SGD(mean.parameters(), lr=0.5)

    class RVStub:
        value = torch.ones_like(param) * 100

    _, tr = mean('z', cond_trace=dict(z=RVStub))
    test = tr['z'].log_prob # (torch.ones_like(param) * 100)

    loss = -test.mean()
    loss.backward()
    #optimizer.step()
    import ipdb; ipdb.set_trace();
    # assert not torch.equal(param, torch.zeros_like(param))




@mark.skip()
def test_gaussian1d():
    target_mean, target_stdv = 4, 1
    num_samples=10
    lr = 1e-3
    proposal = _Gaussian1d(loc=target_stdv, scale=6, name='g_0', num_samples=num_samples)
    target = _Gaussian1d(loc=target_mean, std=1, name='g_1', num_samples=num_samples)
    optimizer = mkAdam([proposal, target], lr=lr)

    num_steps = 100
    loss_ct, loss_sum = 0, 0.0
    loss_all = []
    def trace_eq(t0, t1, name):
        return name in t0 and name in t1 and torch.equal(t0[name].value, t1[name].value)
    def print_traces(ext_only=True):
        def print_grads(tr):
            if tr is not None:
                rvs = ";".join(["{}{}".format(k, "∇" if rv.value.requires_grad else "") for k, rv in tr.items()])
                return f'[{rvs}]'
            else:
                return ""
        if not ext_only:
            try:
                print("p_prv{}: {}".format(print_grads(p_prv), str(p_prv) if p_prv is not None else "None"))
            except:
                print("p_prv: None")
            try:
                print("q_prp{}: {}".format(print_grads(q_prp), str(q_prp) if q_prp is not None else "None"))
            except:
                print("q_prp: None")
        if ext_only:
            try:
                print("q_ext{}: {}".format(print_grads(q_ext), str(q_ext) if q_ext is not None else "None"))
            except:
                print("q_ext: None")
        if not ext_only:
            try:
                print("p_tar{}: {}".format(print_grads(p_tar), str(p_tar) if p_tar is not None else "None"))
            except:
                print("p_tar: None")
        if ext_only:
            try:
                print("p_ext{}: {}".format(print_grads(p_ext), str(p_ext) if p_ext is not None else "None"))
            except:
                print("p_ext: None")

    with trange(num_steps) as bar:
        for i in bar:
            # Initial
            o, p_prv = proposal(  trace_name='z_0')
            # sampling reparameterized will have a gradient
            # NOTE: probtorch will default to running rsample
            lw = torch.zeros([1])
            # ================== #
            # Nestable           #
            # ================== #
            _, q_prp = proposal(trace_name='z_0', cond_trace=p_prv)
            # NOTE: ALL OF p_prv is detached

            _, p_tar = target(trace_name='z_0')


            lv = p_tar.log_joint(batch_dim=None, sample_dims=0, nodes=p_tar._nodes) - \
                q_prp.log_joint(batch_dim=None, sample_dims=0, nodes=p_tar._nodes)

            lw = lw.detach() # completely detached so this is okay with reparam
            values = -lv
            # convice yourself: current intro of JMLR + check with forward grad
            log_weights = torch.zeros_like(lv)
            sample_dims=0
            reducedims=(0,)
            keepdims=False
            # loss = _estimate_mc(-lv, torch.zeros_like(lv), sample_dims=0, reducedims=(0,), keepdims=False,)
            nw = torch.nn.functional.softmax(torch.zeros_like(lv), dim=sample_dims)

            loss = (nw * values).sum(dim=reducedims, keepdim=keepdims)

            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_ct += 1
            loss_scalar = lv.detach().cpu().mean().item()
            loss_sum += loss_scalar
            if num_steps <= 100:
               loss_all.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_all.append(loss_avg)

    with torch.no_grad():
        max_loss = max(loss_all)
        min_loss = min(loss_all)
        print("max: {:.4f}\nmin: {:.4f}".format(max_loss, min_loss))
        print(sparklines.sparklines(list(map(lambda l: l + abs(min_loss), loss_all)))[0])
        # print(sparklines.sparklines(loss_all)[0])

        num_validate_samples = 100
        samples = []
        for _ in range(num_validate_samples):
            _, q_prp = proposal(trace_name='z_0')
            out, q_ext = fwd(q_prp, trace_name='z_1', base_name='z_0')
            # out, _ = proposal()
            samples.append(out)
        evaluation = torch.cat(samples)
        eval_mean, eval_stdv = evaluation.mean().item(), evaluation.std().item()
        print("mean: {:.4f}, std: {:.4f}".format(eval_mean, eval_stdv))
        mean_tol, stdv_tol = 0.25, 0.5
        assert (target_mean - mean_tol) < eval_mean and  eval_mean < (target_mean + mean_tol)
        assert (target_stdv - stdv_tol) < eval_stdv and  eval_stdv < (target_stdv + stdv_tol)
