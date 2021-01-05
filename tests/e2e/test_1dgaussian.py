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

def eval_mean_std(runnable, target_params, tolerances):
    with torch.no_grad():
        num_validate_samples = 400
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
        # sigma = torch.pow(self.sigma(y), 2)
        return mu, torch.ones([self.dim_out]) # *0.5 # sigma

class Gaussian1d(Program):
    def __init__(self, loc:int, std:int, name:str, num_samples:int):
        super().__init__()
        self.loc = loc
        self.std = std
        self.name = name
        self.size = torch.Size([1])
        self.expand_samples = lambda ten: ten.expand(num_samples, *ten.size())

    def model(self, trace, cond_trace=None, num_samples=1):
        return trace.normal(
            loc=self.expand_samples(torch.ones(self.size, requires_grad=True)*self.loc),
            scale=self.expand_samples(torch.ones(self.size)*self.std),
            # value=None if cond_trace is None else cond_trace[trace_name].value,
            name="g")

    def xforward(self, trace_name, cond_trace=None, num_samples=1):
        trace = Trace()
        out = trace.normal(
            loc=self.expand_samples(torch.ones(self.size, requires_grad=True)*self.loc),
            scale=self.expand_samples(torch.ones(self.size)*self.std),
            value=None if cond_trace is None else cond_trace[trace_name].value,

            name=trace_name)
        return out, trace


@typechecked
class SimpleKernel(Kernel):
    def __init__(self, num_hidden, name, ix=None):
        super().__init__()
        self.net = Net(dim_in=1, dim_hidden=num_hidden, dim_out=1)
        self.name = name if ix is None else name + "_" + str(ix)

    def apply_kernel(self, trace, cond_trace, obs):
        mu, scale = self.net(obs.detach())
        # mu, sigma = out[0], out[1]
        return trace.normal(loc=mu, scale=scale, name=self.name)

    def xforward(self, base_trace, base_name, trace_name, cond_trace=None, detach_base=False):
        trace = Trace()

        # INPUT TRACE
        base_rv = base_trace[base_name]
        new_val = base_rv.value.detach() if detach_base else base_rv.value
        # new_val.requires_grad = not detach_base
        copy_rv = RandomVariable(base_rv.dist, new_val, provenance=Provenance.OBSERVED) # ImproperRV in the general case (improper if improper, random if )
        trace.append(copy_rv, name=base_name)

        # Kernel code starts here
        # EXTENDED ARGUMENTS HERE
        mu, sig = self.net(copy_rv.value.detach())
        value = None if cond_trace is None else cond_trace[trace_name].value.detach()
        # import ipdb; ipdb.set_trace();

        out = trace.normal(loc=mu, scale=sig, name=trace_name, value=value)
        # ext_rv = RandomVariable(ext_dist, (ext_dist.sample() if cond_trace is None else cond_trace[trace_name].value), provenance=Provenance.SAMPLED)
        # trace.append(ext_rv, name=trace_name)
        return out, trace

def nvo_avo(lv: Tensor, sample_dims=0) -> Tensor:
    values = -lv
    log_weights = torch.zeros_like(lv)

    nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    loss = (nw * values).sum(dim=(sample_dims,), keepdim=False)
    return loss


@fixture(autouse=True)
def scaffolding():
    torch.manual_seed(1)
    yield

@mark.skip("passes")
def test_program():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    g()

@mark.skip("passes")
def test_forward():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, name="fwd")

    ext = Forward(fwd, g)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

@mark.skip("passes")
def test_reverse():
    g = Gaussian1d(loc=0, std=1, name="g", num_samples=4)
    rev = SimpleKernel(num_hidden=4, name="rev")

    ext = Reverse(g, rev)
    ext()

    for k in ext._cache.program.trace.keys():
        assert torch.equal(ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value)

@mark.skip("passes")
def test_propose():
    q = Gaussian1d(loc=4, std=1, name="q", num_samples=4)
    p = Gaussian1d(loc=0, std=4, name="p", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, name="ext")
    rev = SimpleKernel(num_hidden=4, name="ext")

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    extend = Propose(target=p_ext, proposal=q_ext)

    state, log_weights = extend()

    assert isinstance(log_weights, Tensor)
    assert isinstance(state, PCache) # just a placeholder for the time being

@mark.skip("ideal API")
def test_propose_backprop():
    q = Gaussian1d(loc=0, std=4, name="q", num_samples=4)
    p = Gaussian1d(loc=4, std=1, name="p", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, name="ext")
    rev = SimpleKernel(num_hidden=4, name="ext")

    torch.manual_seed(1)
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [fwd, rev]], lr=0.5)

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)

    extend = Propose(target=p_ext, proposal=q_ext)

    state, log_weights = extend()
    loss = nvo_avo(log_weights, sample_dims=0).mean()
    loss.backward()

    aweight = fwd.net.joint[0].weight.detach()
    optimizer.step()
    after = fwd.net.joint[0].weight.detach()
    assert not torch.equal(aweight, after)
 
def test_forward_reverse():
    q = Gaussian1d(loc=0, std=4, name="q", num_samples=4)
    p = Gaussian1d(loc=4, std=1, name="p", num_samples=4)
    fwd = SimpleKernel(num_hidden=4, name="ext")
    rev = SimpleKernel(num_hidden=4, name="ext")

    torch.manual_seed(1)
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [fwd, rev]], lr=0.5)

    q_ext = Forward(fwd, q)
    p_ext = Reverse(p, rev)
    qtr, _ = q_ext()
    ptr, _ = p_ext()

    log_weights = Propose.log_weights(qtr, ptr)
    loss = nvo_avo(log_weights, sample_dims=0).mean()
    loss.backward()

    aweight = fwd.net.joint[0].weight.detach()
    optimizer.step()
    after = fwd.net.joint[0].weight.detach()
    assert not torch.equal(aweight, after)


@mark.skip()
def test_Gaussian_1step():
    """ The VAE test. At one step no need for any detaches. """
    Params = namedtuple("Params", ["mean", "std", "name"])

    target_params, proposal_params = all_params = [Params(4, 1, "p"), Params(1, 6, "q")]
    target,        proposal        = [Gaussian1d(loc=p.mean, std=p.std, name=p.name, num_samples=200) for p in all_params]
    fwd, rev = [SimpleKernel(num_hidden=4, name=name) for name in ["ext", "ext"]]

    torch.manual_seed(1)
    lr = 1e-2
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [proposal, target, fwd, rev]], lr=lr)

    num_steps = 150
    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []

    with trange(num_steps) as bar:
        for i in bar:
            #=====================================
            # q_prp_trace, _ = proposal()
            # FIXME: for now this is acceptable, but long-term I think thereis a question of
            # 1. inspecting arguments of the proposal and fwd_kernel then compacting the call
            # 2. being able to pull out the trace of the proposal / kernel from forward?
            #   - right now how do you compute logprobs? you need to do this under the trace only, but it would
            #     be nice to just tuck `log_inc_weights` somewhere (maybe on Trace?).
            q_ext = Forward(fwd, proposal)
            qtr, _ = q_ext()
            p_ext = Reverse(target, rev)
            ptr, _ = p_ext()
            import ipdb; ipdb.set_trace();

            extend = Propose(target=p_ext, proposal=q_ext)

            _, log_weights = extend()

            # --------------------------------------------------------------------- #
            loss = nvo_avo(log_weights, sample_dims=0).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

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
        Tolerance = namedtuple("Tolerance", ['mean', 'std'])
        tolerances = Tolerance(mean=0.25, std=0.5)
        eval_mean_std(lambda: Forward(fwd, proposal)()[1], target_params, tolerances=tolerances)


@mark.skip()
def test_Gaussian_2step():
    """
    2-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).

    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
    """
    [prior, target1, target2, final] = [Gaussian1d(i) for i in range(4)]


@mark.skip()
def test_Gaussian_4step():
    """
    3-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).

    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
    """
    [prior, target1, target2, final] = [Gaussian1d(i) for i in range(4)]
    forwards = None
    reverses = None
    targets = None

     #      for r, f, t in zip(forwards, reverses, targets):
     #          p = Propose(Reverse(t, r), Forward(f, p))
     #          out, ptrace, plog_probs, lweight, loss = p()
     #       p = proposal
     #          #    ----------------------------------------------- State
