import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn as nn
from combinators.inference import PCache # temporary
from combinators.stochastic import RandomVariable, Provenance
from probtorch.util import expand_inputs
from collections import namedtuple
from typeguard import typechecked
from tqdm import trange
from pytest import mark, fixture
import sparklines
import combinators.trace.utils as trace_utils
from combinators.tensor.utils import thash, show
from typing import Optional

from combinators import Program, Kernel, Trace, Forward, Reverse, Propose

def report_sparkline(losses:list):
    tlosses = torch.tensor(losses)
    min_loss = tlosses.min().item()
    max_loss = tlosses.max().item()
    print("loss: [max:{:.4f}, min: {:.4f}]".format(max_loss, min_loss))
    baseline = min_loss if min_loss > 0 else -min_loss
    print(sparklines.sparklines((tlosses - baseline).numpy())[0])

Tolerance = namedtuple("Tolerance", ['mean', 'std'])
Params = namedtuple("Params", ["mean", "std"])

def eval_mean_std(runnable, target_params:Params, tolerances:Tolerance, num_validate_samples = 400):
    with torch.no_grad():
        samples = []
        for _ in range(num_validate_samples):
            out = runnable()
            samples.append(out)
        evaluation = torch.cat(samples)
        eval_mean, eval_stdv = evaluation.mean().item(), evaluation.std().item()
        print("mean: {:.4f}, std: {:.4f}".format(eval_mean, eval_stdv))
        assert (target_params.mean - tolerances.mean) < eval_mean and  eval_mean < (target_params.mean + tolerances.mean)
        assert (target_params.std  - tolerances.std ) < eval_stdv and  eval_stdv < (target_params.std  + tolerances.std )


class Net(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out))
        # self.sigma = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        y = self.joint(x)
        mu = self.mu(y)
        # sigma = torch.pow(self.sigma(y), 2) / 4 # keep things positive but also scale it down and keep things opinionated
        return mu, torch.ones([self.dim_out]) # This works, but test sparklines become less pretty

class Gaussian1d(Program):
    def __init__(self, loc:int, std:int, name:str, num_samples:int):
        super().__init__()
        self.loc = loc
        self.std = std
        self.name = name
        self.size = torch.Size([1])
        self.expand_samples = lambda ten: ten.expand(num_samples, *ten.size())

    def model(self, trace):
        return trace.normal(
            loc=self.expand_samples(torch.ones(self.size, requires_grad=True)*self.loc),
            scale=self.expand_samples(torch.ones(self.size)*self.std),
            # value=None if cond_trace is None else cond_trace[trace_name].value,
            name=self.name)

    def __repr__(self):
        return f"Gaussian1d(loc={self.loc}, scale={self.std})"

@typechecked
class SimpleKernel(Kernel):
    def __init__(self, num_hidden, ext_name):
        super().__init__()
        self.net = Net(dim_in=1, dim_hidden=num_hidden, dim_out=1)
        self.ext_name = ext_name

    def apply_kernel(self, trace, cond_trace, obs):
        mu, scale = self.net(obs.detach())
        return trace.normal(loc=mu, scale=scale, name=self.ext_name)

def nvo_avo(lv: Tensor, sample_dims=0) -> Tensor:
    values = -lv
    log_weights = torch.zeros_like(lv)

    nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    loss = (nw * values).sum(dim=(sample_dims,), keepdim=False)
    return loss


@fixture(autouse=True)
def scaffolding():
    torch.manual_seed(1)


def test_program():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    g()

def test_forward():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, ext_name="fwd")

    ext = Forward(fwd, g)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

def test_forward_forward():
    g0 = Gaussian1d(loc=0, std=1, name="g0", num_samples=4)
    f01 = SimpleKernel(num_hidden=4, ext_name="g1")
    f12 = SimpleKernel(num_hidden=4, ext_name="g2")

    ext = Forward(f12, Forward(f01, g0))
    ext()


def test_reverse():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    rev = SimpleKernel(num_hidden=4, ext_name="rev")

    ext = Reverse(g, rev)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

def test_propose_values():
    q = Gaussian1d(loc=4, std=1, name="q", num_samples=4)
    p = Gaussian1d(loc=0, std=4, name="p", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, ext_name="p")
    rev = SimpleKernel(num_hidden=4, ext_name="q")
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5)
    assert len(list(p.parameters())) == 0
    assert len(list(q.parameters())) == 0
    fwd_hashes_0 = [thash(t) for t in fwd.parameters()]
    rev_hashes_0 = [thash(t) for t in rev.parameters()]

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    extend = Propose(target=p_ext, proposal=q_ext)

    _, log_weights = extend()

    assert isinstance(log_weights, Tensor)

    cache = extend._cache
    # import ipdb; ipdb.set_trace();

    for k in ['p', 'q']:
        assert torch.equal(cache.proposal.trace[k].value, cache.target.trace[k].value)

    loss = nvo_avo(log_weights, sample_dims=0).mean()
    loss.backward()

    optimizer.step()
    fwd_hashes_1 = [thash(t) for t in fwd.parameters()]
    rev_hashes_1 = [thash(t) for t in rev.parameters()]

    assert any([l != r for l, r in zip(fwd_hashes_0, fwd_hashes_1)])
    assert any([l != r for l, r in zip(rev_hashes_0, rev_hashes_1)])

# @mark.skip("this is not a problem anymore because of the observation clearing -- not sure if this is a problem yet...")
def test_propose_gradients():
    q = Gaussian1d(loc=4, std=1, name="q", num_samples=4)
    p = Gaussian1d(loc=0, std=4, name="p", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, ext_name="p")
    rev = SimpleKernel(num_hidden=4, ext_name="q")
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5)

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    extend = Propose(target=p_ext, proposal=q_ext)

    _, log_weights = extend()
    cache = extend._cache

    for k, prg in [('p', cache.target), ('q', cache.target), ('p', cache.proposal)]:
        assert prg.trace[k].value.requires_grad

    assert not cache.proposal.trace['q'].value.requires_grad


def test_Gaussian_1step():
    """ The VAE test. At one step no need for any detaches. """
    Params = namedtuple("Params", ["mean", "std", "name"])

    target_params, proposal_params = all_params = [Params(4, 1, "p"), Params(1, 4, "q")]
    target,        proposal        = [Gaussian1d(loc=p.mean, std=p.std, name=p.name, num_samples=200) for p in all_params]
    fwd, rev = [SimpleKernel(num_hidden=4, ext_name=ext_name) for ext_name in ["p", "q"]]

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [proposal, target, fwd, rev]], lr=0.01)

    num_steps = 100
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            # tr_q[q]   , out_q = proposal()
            # tr_f[q, p], out_f = fwd(tr_q, out_q)  ==> tr_f[q] == tr_q[q]
            q_ext = Forward(fwd, proposal) # <- this is a program
            # tr_qext[q, p], out_qext = q_ext() => tr_qext[q, p] == tr_f[q, p]

            p_ext = Reverse(target, rev)
            extend = Propose(target=p_ext, proposal=q_ext)

            _, log_weights = extend()

            loss = nvo_avo(log_weights, sample_dims=0).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if num_steps <= 100:
               loss_avgs.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_avgs.append(loss_avg)
    with torch.no_grad():
        report_sparkline(loss_avgs)
        eval_mean_std(lambda: Forward(fwd, proposal)()[1], target_params, Tolerance(mean=0.15, std=0.15))


def test_Gaussian_2step():
    """
    2-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).

    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
    """
    g1, g2, g3 = targets = [Gaussian1d(loc=i, std=1, name=f"z_{i}", num_samples=200) for i in range(1,4)]
    f12, f23 = forwards = [SimpleKernel(num_hidden=4, ext_name=f"z_{i}") for i in range(2,4)]
    r21, r32 = reverses = [SimpleKernel(num_hidden=4, ext_name=f"z_{i}") for i in range(1,3)]

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=1e-2)

    num_steps = 1000
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q0 = targets[0]
            q_prv_tr, _ = q0()
            losses = []

            debug_states = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                q.condition_on(q_prv_tr)
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                state, log_weights = extend()
                debug_states.append(state)
                q_prv_tr = state.proposal.trace
                losses.append(nvo_avo(log_weights, sample_dims=0).mean())

            loss = torch.vstack(losses).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if num_steps <= 100:
               loss_avgs.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_avgs.append(loss_avg)
    with torch.no_grad():
        report_sparkline(loss_avgs)

        predict_g2 = lambda: Forward(f12, g1)()[1]
        predict_g3 = lambda: Forward(f23, Forward(f12, g1))()[1]
        import ipdb; ipdb.set_trace();

        tol = Tolerance(mean=0.15, std=0.15)
        eval_mean_std(predict_g2, Params(mean=2, std=1), tol)
        eval_mean_std(predict_g3, Params(mean=3, std=1), tol)


def test_Gaussian_4step():
    """
    4-step NVI-sequential: 8 intermediate densities
    """
    ix0 = 1 # first target index (also first target's mean)
    ixF = 5 # final target index (also final target's mean)
    g1, g2, g3, g4, g5 = targets = [Gaussian1d(loc=i, std=1, name=f"z_{i}", num_samples=200) for i in range(1,6)]
    f12, f23, f34, f45 = forwards = [SimpleKernel(num_hidden=4, ext_name=f"z_{i}") for i in range(2,6)]
    r21, r32, r43, r54 = reverses = [SimpleKernel(num_hidden=4, ext_name=f"z_{i}") for i in range(1,5)]
    assert r21.ext_name == "z_1"
    assert f12.ext_name == "z_2"
    assert r54.ext_name == "z_4"
    assert f45.ext_name == "z_5"

    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]], lr=0.01)

    num_steps = 200
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            q_prev = targets[0]
            losses = []
            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
                q_ext = Forward(fwd, q)
                p_ext = Reverse(p, rev)
                extend = Propose(target=p_ext, proposal=q_ext)
                _, log_weights = extend()
                losses.append(nvo_avo(log_weights, sample_dims=0).mean())

            loss = torch.vstack(losses).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # REPORTING
            loss_ct += 1
            loss_scalar = loss.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_all.append(loss_scalar)
            if num_steps <= 100:
               loss_avgs.append(loss_scalar)
            if i % 10 == 0:
               loss_avg = loss_sum / loss_ct
               loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
               bar.set_postfix_str(loss_template)
               loss_ct, loss_sum  = 0, 0.0
               if num_steps > 100:
                   loss_avgs.append(loss_avg)
    with torch.no_grad():
        report_sparkline(loss_avgs)
        tol = Tolerance(mean=0.15, std=0.15)

        predict_g2_chain = lambda: Forward(f12, g1)()[1]
        predict_g3_chain = lambda: Forward(f23, Forward(f12, g1))()[1]
        predict_g4_chain = lambda: Forward(f34, Forward(f23, Forward(f12, g1)))()[1]
        predict_g5_chain = lambda: Forward(f45, Forward(f34, Forward(f23, Forward(f12, g1))))()[1]

        eval_mean_std(predict_g2_chain, Params(mean=2, std=1), tol)
        eval_mean_std(predict_g3_chain, Params(mean=3, std=1), tol)
        eval_mean_std(predict_g4_chain, Params(mean=4, std=1), tol)
        eval_mean_std(predict_g5_chain, Params(mean=5, std=1), tol)

        predict_g2 = lambda: Forward(f12, g1)()[1]
        predict_g3 = lambda: Forward(f23, g2)()[1]
        predict_g4 = lambda: Forward(f34, g3)()[1]
        predict_g5 = lambda: Forward(f45, g4)()[1]

        eval_mean_std(predict_g2, Params(mean=2, std=1), tol)
        eval_mean_std(predict_g3, Params(mean=3, std=1), tol)
        eval_mean_std(predict_g4, Params(mean=4, std=1), tol)
        eval_mean_std(predict_g5, Params(mean=5, std=1), tol)
