#!/usr/bin/env python
#
import torch
import torch.nn as nn
import logging
from torch import Tensor
from torch import distributions
from torch.utils.tensorboard import SummaryWriter
from pytest import fixture, mark
from tqdm import trange

import combinators.trace.utils as trace_utils
from combinators.trace.utils import RequiresGrad
from combinators.tensor.utils import autodevice, kw_autodevice, copy, thash
from combinators import Forward, Reverse, Propose, Kernel, Condition
from combinators.metrics import effective_sample_size, log_Z_hat
from combinators.debug import propagate, print_grads, propagate

from combinators.nnets import LinearMap, ResMLPJ
from combinators.objectives import nvo_rkl, nvo_avo
from combinators.densities import MultivariateNormal, Tempered, RingGMM
from combinators.densities.kernels import MultivariateNormalLinearKernel, MultivariateNormalKernel, IdentityKernel
from tests.utils import is_smoketest, seed


def test_tempered_redundant_noloop(seed):
    # hyperparams
    sample_shape = (13,)

    # Setup
    g0 = MultivariateNormal(name='g0', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())**2)
    gK = MultivariateNormal(name='g2', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())**2)
    halfway = Tempered('g1', g0, gK, torch.tensor([0.5], **kw_autodevice() ))
    f01, f12 = [MultivariateNormalLinearKernel(ext_from=f'g{i}', ext_to=f'g{i+1}', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())).to(autodevice()) for i in range(0,2)]
    r10, r21 = [MultivariateNormalLinearKernel(ext_from=f'g{i+1}', ext_to=f'g{i}', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())).to(autodevice()) for i in range(0,2)]

    # NVI labels:
    p_prv_tr, _, _ = g0(sample_shape=sample_shape)
    q1_ext = Forward(f01, g0)
    p1_ext = Reverse(halfway, r10)
    extend1 = Propose(target=p1_ext, proposal=q1_ext)
    state1 = extend1(sample_shape=sample_shape, sample_dims=0)

    q2_ext = Forward(f12, Condition(halfway, state1.trace))
    p2_ext = Reverse(gK, r21)
    extend2 = Propose(target=p2_ext, proposal=q2_ext)
    state2 = extend2(sample_shape=sample_shape, sample_dims=0)

def test_tempered_redundant_loop(seed, is_smoketest):
    # hyperparams
    sample_shape = (100,)
    num_iterations = 10 if is_smoketest else 1000

    # Setup
    g0 = MultivariateNormal(name='g0', loc=torch.ones(2, **kw_autodevice()),   cov=torch.eye(2, **kw_autodevice())**2)
    gK = MultivariateNormal(name='g2', loc=torch.ones(2, **kw_autodevice())*3, cov=torch.eye(2, **kw_autodevice())**2)
    halfway = Tempered('g1', g0, gK, torch.tensor([0.5], **kw_autodevice()))
    forwards = [MultivariateNormalLinearKernel(ext_from=f'g{i}', ext_to=f'g{i+1}', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())).to(autodevice()) for i in range(0,2)]
    reverses = [MultivariateNormalLinearKernel(ext_from=f'g{i+1}', ext_to=f'g{i}', loc=torch.ones(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())).to(autodevice()) for i in range(0,2)]
    targets = [g0, halfway, gK]
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-4)

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_iterations) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, _, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros(1, **kw_autodevice())
            lw, lvs = torch.zeros(sample_shape, **kw_autodevice()), []
            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q_ext = Forward(fwd, Condition(q, p_prv_tr), _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state = extend(sample_shape=sample_shape, sample_dims=0)
                lv = state.log_joint

                p_prv_tr = state.trace

                lw += lv
                loss += nvo_rkl(lw, lv, extend.proposal._cache.trace[f'g{k}'], state.trace[f'g{k+1}'])
                lvs.append(lv)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            writer.add_scalar('Loss/train', loss_scalar, i)
            loss_sum += loss_scalar
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               loss_ct, loss_sum  = 0, 0.0
               bar.set_postfix_str(loss_template)


    with torch.no_grad():
        f12, f23 = forwards
        analytic2 = propagate(N=g0.dist, F=f12.weight(), t=f12.bias(), B=g0.dist.covariance_matrix, marginalize=True)
        # TODO: add some tests?


def test_annealing_path_tempered_normals(seed, is_smoketest):
    num_targets = 3
    g0 = MultivariateNormal(name='g0', loc=torch.zeros(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())**2)
    gK = MultivariateNormal(name=f'g{num_targets}', loc=torch.ones(2, **kw_autodevice())*num_targets, cov=torch.eye(2, **kw_autodevice())**2)
    forwards = [MultivariateNormalLinearKernel(ext_from=f'g{i}',ext_to=f'g{i+1}', loc=torch.ones(2)*i, cov=torch.eye(2)).to(autodevice()) for i in range(num_targets)]
    reverses = [MultivariateNormalLinearKernel(ext_from=f'g{i+1}',ext_to=f'g{i}', loc=torch.ones(2)*i, cov=torch.eye(2)).to(autodevice()) for i in range(num_targets)]

    betas = torch.arange(0., 1., 1./num_targets, **kw_autodevice())[1:] # g_0 is beta=0

    path = [Tempered(f'g{k}', g0, gK, beta) for k, beta in zip(range(1,num_targets), betas)]
    path = [g0] + path + [gK]
    targets = path

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-3)

    num_steps = 10 if is_smoketest else 1000
    sample_shape = (100,)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr, _, _ = q0(sample_shape=sample_shape)
            loss = torch.zeros(1, **kw_autodevice())

            lvss = []
            lw = torch.zeros(sample_shape, **kw_autodevice())

            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q_ext = Forward(fwd, Condition(q, p_prv_tr, requires_grad=RequiresGrad.NO))
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state = extend(sample_shape=sample_shape, sample_dims=0)
                lv = state.log_joint
                lvss.append(lv.detach())

                # FIXME: because p_prv_tr is not eliminating the previous trace, the trace is cumulativee but removing grads leaves backprop unaffected
                p_prv_tr = state.trace
                assert set(p_prv_tr.keys()) == { f'g{k+1}' }

                lw += lv
                # loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.trace[f'g{k+1}'])
                if k==2:
                    loss = nvo_avo(lv)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                # REPORTING
                # ---------------------------------------
                # loss
                loss_ct += 1
                loss_scalar = loss.detach().cpu().mean().item()
                # writer.add_scalar('loss', loss_scalar, i)
                loss_sum += loss_scalar

                # ESS
                lvs = torch.stack(lvss, dim=0)
                lws = torch.cumsum(lvs, dim=1)
                ess = effective_sample_size(lws, sample_dims=-1)
                # for step, x in zip(range(1,len(ess)+1), ess):
                #     writer.add_scalar(f'ess/step-{step}', x, i)

                # logZhat
                lzh = log_Z_hat(lws, sample_dims=-1)
                # for step, x in zip(range(1,len(lzh)+1), lzh):
                #     writer.add_scalar(f'log_Z_hat/step-{step}', x, i)

                # progress bar
                if i % 10 == 0:
                    loss_avg = loss_sum / loss_ct
                    loss_template = 'loss={:3.4f}'.format(loss_avg)
                    lZh_template = "lZh:" + ";".join(['{:3.2f}'.format(lzh[i].cpu().item()) for i in range(len(lzh))])
                    ess_template = "ess:" + ";".join(['{:3.1f}'.format(ess[i].cpu().item()) for i in range(len(ess))])
                    loss_ct, loss_sum  = 0, 0.0
                    bar.set_postfix_str("; ".join([loss_template, ess_template, lZh_template]))

    with torch.no_grad():
        fp011, fp121, fp231 = fps1 = [[copy(p, requires_grad=False) for p in fwd.parameters()] for fwd in reverses]

        hashes_fps1 = [thash(f) for fs in fps1 for f in fs]

        empirical_loc1 = Forward(forwards[0], targets[0])(sample_shape=(2000,)).output.mean(dim=0)
        analytic1 = propagate(N=targets[0].dist, F=forwards[0].weight(), t=forwards[0].bias(), B=targets[0].dist.covariance_matrix, marginalize=True)
        print(empirical_loc1, analytic1)
        empirical_loc2 = Forward(forwards[1], Forward(forwards[0], targets[0]))(sample_shape=(2000,)).output.mean(dim=0)
        print(empirical_loc2)
        empirical_loc3 = Forward(forwards[2], Forward(forwards[1], Forward(forwards[0], targets[0])))(sample_shape=(2000,)).output.mean(dim=0)
        print(empirical_loc3)


def mk_kernel(target:int, std:float, num_hidden:int):
    embedding_dim = 2
    return MultivariateNormalKernel(
        ext_from=f'g_{target-1}',
        ext_to=f'g_{target}',
        loc=torch.zeros(2),
        cov=torch.eye(2)*std**2,
        net=ResMLPJ(
            dim_in=2,
            dim_hidden=num_hidden,
            dim_out=embedding_dim,
        ))

def mk_model(num_targets:int):
    proposal_std = 1.0
    g_0 = MultivariateNormal(name='g_0', loc=torch.zeros(2), cov=torch.eye(2)*proposal_std**2)
    g_K = RingGMM(scale=8, count=8, name=f"g_{num_targets - 1}")

    # Make an annealing path
    betas = torch.arange(0., 1., 1./(num_targets - 1))[1:] # g_0 is beta=0
    path = [Tempered(f'g_{k}', g_0, g_K, beta) for k, beta in zip(range(1,num_targets-1), betas)]
    path = [g_0] + path + [g_K]
    assert len(path) == num_targets # sanity check that the betas line up

    num_kernels = num_targets - 1
    target_ixs = [ix for ix in range(0, num_targets)]
    mk_kernels = lambda shift_target: [
        mk_kernel(target=shift_target(ix),
                  std=shift_target(ix)+1.0,
                  num_hidden=64
                  ) for ix in target_ixs[:num_kernels]
    ]

    return dict(
        targets=path,
        forwards=mk_kernels(lambda ix: ix+1),
        reverses=mk_kernels(lambda ix: ix),
    )

@mark.skip("right before the big one")
def test_annealing_path_8step_simple(seed):
    K = 8
    out = mk_model(K)
    targets, forwards, reverses = [out[n] for n in ['targets', 'forwards', 'reverses']]
    assert all([len(list(t.parameters())) == 0 for t in targets])
    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)
    num_samples = 7
    num_steps = 13
    sample_shape=(num_samples,)
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            p_prv_tr = q0(sample_shape=sample_shape)[0]
            loss = torch.zeros(1)

            lvs = []
            lw = torch.zeros(sample_shape)

            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q_ext = Forward(fwd, Condition(q, p_prv_tr), _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
                state = extend(sample_shape=sample_shape, sample_dims=0)
                lv = state.log_joint

                # FIXME: because p_prv_tr is not eliminating the previous trace, the trace is cumulativee but removing grads leaves backprop unaffected
                p_prv_tr = state.trace
                assert set(p_prv_tr.keys()) == { f'g_{k+1}' }

                lw += lv
                loss += nvo_rkl(lw, lv, state.proposal.trace[f'g_{k}'], state.trace[f'g_{k+1}'])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0


@mark.skip("this is the big one")
def test_tempered_grad_check(seed):
    #!/usr/bin/env python3
    import torch
    import math
    from torch import nn, Tensor
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import trange
    from typing import Tuple
    from matplotlib import pyplot as plt

    import combinators.trace.utils as trace_utils
    from combinators.trace.utils import RequiresGrad
    from combinators.tensor.utils import autodevice, kw_autodevice, copy, show
    from combinators.densities import MultivariateNormal, Tempered, RingGMM, Normal
    from combinators.densities.kernels import MultivariateNormalKernel, MultivariateNormalLinearKernel, NormalLinearKernel
    from combinators.nnets import ResMLPJ
    from combinators.objectives import nvo_rkl, nvo_avo
    from combinators import Forward, Reverse, Propose
    from combinators.stochastic import RandomVariable, ImproperRandomVariable
    from combinators.metrics import effective_sample_size, log_Z_hat
    # import visualize as V

    def mk_kernel(from_:int, to_:int, std:float, num_hidden:int, learn_cov=True):
        embedding_dim = 2
        return MultivariateNormalKernel(
            ext_from=f'g{from_}',
            ext_to=f'g{to_}',
            loc=torch.zeros(2, **kw_autodevice()),
            cov=torch.eye(2, **kw_autodevice())*std**2,
            learn_cov=learn_cov,
            net=ResMLPJ(
                dim_in=2,
                dim_hidden=num_hidden,
                dim_out=embedding_dim).to(autodevice()))

    def mk_mnlinear_kernel(from_:int, to_:int, std:float, dim:int):
        return MultivariateNormalLinearKernel(
            ext_from=f'g{from_}',
            ext_to=f'g{to_}',
            loc=torch.zeros(dim, **kw_autodevice()),
            cov=torch.eye(dim, **kw_autodevice())*std**2)

    def mk_nlinear_kernel(from_:int, to_:int, std:float, dim:int):
        return NormalLinearKernel(ext_from=f'g{from_}', ext_to=f'g{to_}')

    def mk_targets(num_targets):
        proposal_std = 8
        g0 = MultivariateNormal(name='g0', loc=torch.zeros(2, **kw_autodevice()), cov=torch.eye(2, **kw_autodevice())*4**2)
        gK = MultivariateNormal(name=f"g{num_targets - 1}", loc=torch.ones(2, **kw_autodevice())*4, cov=torch.eye(2, **kw_autodevice())**2)

        # Make an annealing path
        betas = torch.arange(0., 1., 1./(num_targets - 1))[1:] # g_0 is beta=0
        path = [Tempered(f'g{k}', g0, gK, beta) for k, beta in zip(range(1,num_targets-1), betas)]
        path = [g0] + path + [gK]
        assert len(path) == num_targets # sanity check that the betas line up
        return path

    def anneal_between(left, right, total_num_targets):
        proposal_std = total_num_targets

        # Make an annealing path
        betas = torch.arange(0., 1., 1./(total_num_targets - 1))[1:] # g_0 is beta=0
        path = [Tempered(f'g{k}', left, right, beta) for k, beta in zip(range(1,total_num_targets-1), betas)]
        path = [left] + path + [right]

        assert len(path) == total_num_targets # sanity check that the betas line up
        return path


    def anneal_between_mvns(left_loc, right_loc, total_num_targets):
        g0 = mk_mvn(0, left_loc, scale=4)
        gK =  mk_mvn(total_num_targets-1, right_loc)

        return anneal_between(g0, gK, total_num_targets)

    def anneal_between_ns(left_loc, right_loc, total_num_targets):
        g0 = mk_n(0, left_loc)
        gK =  mk_n(total_num_targets-1, right_loc)

        return anneal_between(g0, gK, total_num_targets)

    def mk_mvn(i, loc, scale=1):
        return MultivariateNormal(name=f'g{i}', loc=torch.ones(2, **kw_autodevice())*loc, cov=torch.eye(2, **kw_autodevice())*scale**2)

    def mk_n(i, loc):
        return Normal(name=f'g{i}', loc=torch.ones(1, **kw_autodevice())*loc, scale=torch.ones(1, **kw_autodevice())**2)

    def mk_model(num_targets:int):
        return dict(
    #         targets=mk_targets(num_targets),
    #         forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets-1)],
    #         reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets-1)],

     #        targets=anneal_between_mvns(0, num_targets*2, num_targets),
     #        forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=64) for i in range(num_targets-1)],
     #        reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=64) for i in range(num_targets-1)],

            targets=anneal_between_mvns(0, num_targets*2, num_targets),
            forwards=[mk_mnlinear_kernel(from_=i, to_=i+1, std=1., dim=2) for i in range(num_targets-1)],
            reverses=[mk_mnlinear_kernel(from_=i+1, to_=i, std=1., dim=2) for i in range(num_targets-1)],

            # targets=anneal_between_ns(0, num_targets*2, num_targets),
            # forwards=[mk_nlinear_kernel(from_=i, to_=i+1, std=1., dim=1) for i in range(num_targets-1)],
            # reverses=[mk_nlinear_kernel(from_=i+1, to_=i, std=1., dim=1) for i in range(num_targets-1)],

     #        targets=[mk_mvn(i, i*2) for i in range(num_targets)],
     #        forwards=[mk_kernel(from_=i, to_=i+1, std=1., num_hidden=32) for i in range(num_targets-1)],
     #        reverses=[mk_kernel(from_=i+1, to_=i, std=1., num_hidden=32) for i in range(num_targets-1)],

      #       targets=[mk_mvn(i, i*2) for i in range(num_targets)],
      #       forwards=[mk_mnlinear_kernel(from_=i, to_=i+1, std=1., dim=2) for i in range(num_targets-1)],
      #       reverses=[mk_mnlinear_kernel(from_=i+1, to_=i, std=1., dim=2) for i in range(num_targets-1)],

    #         targets=[mk_n(i, i*2) for i in range(num_targets)],
    #         forwards=[mk_nlinear_kernel(from_=i, to_=i+1, std=1., dim=1) for i in range(num_targets-1)],
    #         reverses=[mk_nlinear_kernel(from_=i+1, to_=i, std=1., dim=1) for i in range(num_targets-1)],
        )
    K = 5
    mk_model(K)
    import torch
    import math
    from torch import nn, Tensor
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import trange
    from typing import Tuple
    from matplotlib import pyplot as plt

    import combinators.trace.utils as trace_utils
    from combinators.tensor.utils import autodevice, kw_autodevice
    from combinators.densities import MultivariateNormal, Tempered, RingGMM
    from combinators.densities.kernels import MultivariateNormalKernel
    from combinators.nnets import ResMLPJ
    from combinators.objectives import nvo_rkl
    from combinators import Forward, Reverse, Propose
    from combinators.stochastic import RandomVariable, ImproperRandomVariable
    from combinators.metrics import effective_sample_size, log_Z_hat
    # import visualize as V
    from combinators import Forward

    def sample_along(proposal, kernels, sample_shape=(2000,)):
        samples = []
        tr, out = proposal(sample_shape=sample_shape)
        samples.append(out)
        for k in forwards:
            proposal = Forward(k, proposal)
            tr, out = proposal(sample_shape=sample_shape)
            samples.append(out)
        return samples
    # main() arguments
    seed=1
    eval_break = 500
    # Setup
    torch.manual_seed(seed)
    # K = 4
    num_samples = 256
    sample_shape=(num_samples,)

    # Models
    out = mk_model(K)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    assert all([len(list(k.parameters())) >  0 for k in [*forwards, *reverses]])
    print(targets)

    print("<><><><><><><><><><><><><><><><><><>")
    torch.manual_seed(6)
    samples = sample_along(targets[0], forwards)
    plot_type = len(samples[0].squeeze().shape)
    print('quick empirical')

    if plot_type == 1:
        print(";  ".join(["{:.4f}".format(ss.mean().cpu().item()) for ss in samples]))
    elif plot_type == 2:
        print(";  ".join(["{:.4f}x{:.4f}".format(ss[:,0].mean().cpu().item(), ss[:,1].mean().cpu().item()) for ss in samples]))
    print("<><><><><><><><><><><><><><><><><><>")

    # logging
    writer = SummaryWriter()
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []
    from combinators.objectives import mb0, mb1, _estimate_mc, eval_nrep
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=1e-2)
    # optimizer = torch.optim.SGD([dict(params=x.parameters()) for x in [*forwards, *reverses]], lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    num_iterations=500
    fp0 = [[p.detach().clone() for p in fwd.parameters()] for fwd in forwards]
    rp0 = [[p.detach().clone() for p in fwd.parameters()] for fwd in reverses]

    # print(forwards[0])
    # print(reverses[0])
    # print("=====================================")
    #
    # _ = [print(p) for f in forwards  for p in f.parameters()]
    # printer = lambda forwards: [print(p) for f in forwards  for p in f.parameters()]
    #
    # def compare(ks0, ks1):
    #     params_all = []
    #     for i in range(len(ks0)):
    #         params_cmp = []
    #         for j in range(len(ks0[i])):
    #             params_cmp.append(torch.equal(ks0[i][j], ks1[i][j]))
    #         params_all.append(params_cmp)
    #     return params_all
    #
    # print("-------------------------------------")
    # _ = [print(p) for f in reverses for p in f.parameters()]
    with trange(num_iterations) as bar:
        for i in bar:
            optimizer.zero_grad()
            q0 = targets[0]
            p_prv_tr, out0 = q0(sample_shape=sample_shape)

            loss = torch.zeros(1, **kw_autodevice())
            lw, lvss = torch.zeros(sample_shape, **kw_autodevice()), []
            for k, (fwd, rev, q, p) in enumerate(zip(forwards, reverses, targets[:-1], targets[1:])):
                q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
                q_ext = Forward(fwd, q, _step=k)
                p_ext = Reverse(p, rev, _step=k)
                extend = Propose(target=p_ext, proposal=q_ext, _step=k)
    #             breakpoint()
                state = extend(sample_shape=sample_shape, sample_dims=0)
                lv = state.log_joint

                p_prv_tr = state.trace
                p.clear_observations()
                q.clear_observations()
                lw += lv
    #             loss += nvo_avo(lv)
    #             loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.trace[f'g{k+1}'])

                # batch_dim=None
                # sample_dims=0
                # rv_proposal=state.proposal.trace[f'g{k}']
                # rv_target=state.trace[f'g{k+1}']
                # # TODO: move back from the proposal and target RVs to joint logprobs?
                # reducedims = (sample_dims,)
                #
                # lw = lw.detach()
                # ldZ = lv.detach().logsumexp(dim=sample_dims) - math.log(lv.shape[sample_dims])
                # f = -lv
                #
                # # rv_proposal = next(iter(proposal_trace.values())) # tr[\gamma_{k-1}]
                # # rv_target = next(iter(target_trace.values()))     # tr[\gamma_{k}]
                #
                # kwargs = dict(
                #     sample_dims=sample_dims,
                #     reducedims=reducedims,
                #     keepdims=False
                # )
                #
                # baseline = _estimate_mc(f.detach(), lw, **kwargs).detach()
                #
                # kl_term = _estimate_mc(mb1(rv_proposal.log_prob.squeeze()) * (f - baseline), lw, **kwargs)
                #
                # grad_log_Z1 = _estimate_mc(rv_proposal.log_prob.squeeze(), lw, **kwargs)
                # grad_log_Z2 = _estimate_mc(eval_nrep(rv_target).log_prob.squeeze(), lw+lv.detach(), **kwargs)

                if k==2:
                    # loss += kl_term + mb0(baseline * grad_log_Z1 - grad_log_Z2) + baseline + ldZ
                    loss += nvo_rkl(lw, lv, state.proposal.trace[f'g{k}'], state.trace[f'g{k+1}'])
                    # loss += nvo_avo(lv)


                lvss.append(lv)

            loss.backward()
            optimizer.step()
            # fp1 = [[p.detach().clone() for p in fwd.parameters()] for fwd in forwards]
            # rp1 = [[p.detach().clone() for p in fwd.parameters()] for fwd in reverses]
            # cf = lambda: compare(fp0, fp1)
            # cr = lambda: compare(rp0, rp1)
            # print(cf())
            # print(cr())
            # breakpoint()

    #         scheduler.step()

            # with torch.no_grad():
            # REPORTING
            # ---------------------------------------
    #         # ESS
            lvs = torch.stack(lvss, dim=0)
            lws = torch.cumsum(lvs, dim=1)
            ess = effective_sample_size(lws, sample_dims=-1)
            # for step, x in zip(range(1,len(ess)+1), ess):
            #     writer.add_scalar(f'ess/step-{step}', x, i)

            # logZhat
            lzh = log_Z_hat(lws, sample_dims=-1)
            # for step, x in zip(range(1,len(lzh)+1), lzh):
            #     writer.add_scalar(f'log_Z_hat/step-{step}', x, i)

            # loss
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            writer.add_scalar('loss', loss_scalar, i)
            loss_sum += loss_scalar

            # progress bar
            if i % 10 == 0:
                loss_avg = loss_sum / loss_ct
                loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
                logZh_template = 'logZhat[-1]={:.4f}'.format(lzh[-1].cpu().item())
                ess_template = 'ess[-1]={:.4f}'.format(ess[-1].cpu().item())
                loss_ct, loss_sum  = 0, 0.0
                bar.set_postfix_str("; ".join([loss_template, ess_template, logZh_template]))

                # show samples
    #             if i % (eval_break + 1) == 0:
    #                 samples = sample_along(targets[0], forwards)
    #                 fig = V.scatter_along(samples)
    #                 writer.add_figure('overview', fig, global_step=i, close=True)
    # #                 for ix, xs in enumerate(samples):
    # #                     writer.add_figure(f'step-{ix+1}', V.scatter(xs), global_step=i, close=True)
    print("<><><><><><><><><><><><><><><><><><>")
    torch.manual_seed(6)
    samples = sample_along(targets[0], forwards)
    plot_type = len(samples[0].squeeze().shape)
    print('quick empirical')

    if plot_type == 1:
        print(";  ".join(["{:.4f}".format(ss.mean().cpu().item()) for ss in samples]))
    elif plot_type == 2:
        print(";  ".join(["{:.4f}x{:.4f}".format(ss[:,0].mean().cpu().item(), ss[:,1].mean().cpu().item()) for ss in samples]))
    print("<><><><><><><><><><><><><><><><><><>")


    print("=====================================")

    _ = [print(p) for f in forwards  for p in f.parameters()]
    print("-------------------------------------")

    _ = [print(p) for f in reverses for p in f.parameters()]
    breakpoint();
    print()
