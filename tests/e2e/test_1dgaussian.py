import torch
from torch import Tensor
import torch.nn as nn
from combinators import Program, Kernel, Forward, Reverse
from typeguard import typechecked
from tqdm import trange
from pytest import mark

class Net(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(dim_hidden, dim_out))
        self.sigma = nn.Sequential(nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        y = self.joint(x)
        mu = self.mu(y) + x
        return mu
        # Save stdev for later
        # sigma = self.sigma(y)
        # return torch.cat((mu, sigma), dim=-1)

class Gaussian1d(Program):
    def __init__(self, loc:int, name:str):
        super().__init__()
        self.loc = loc
        self.name = name
        self.size = torch.Size([1])

    def model(self, trace):
        return trace.normal(loc=torch.ones(self.size) *self.loc, scale=torch.ones(self.size), name=self.name)

@typechecked
class SimpleKernel(Kernel):
    def __init__(self, num_hidden, name, ix=None):
        super().__init__()
        self.net = Net(dim_in=1, dim_hidden=num_hidden, dim_out=1)
        self.name = name if ix is None else name + "_" + str(ix)

    def apply_kernel(self, trace, cond_trace, obs):
        mu = self.net(obs)
        # mu, sigma = out[0], out[1]
        return trace.normal(loc=mu, scale=torch.ones([1]), name=self.name)

def objective(lvs, lws):
    """ important for n > 1 where you need to detach when finding the normalizing constant """
    nws = nn.functional.softmax(lws.detach(), dim=0)
    return -(nws * lvs).sum(dim=0)

def test_Gaussian_1step():
    """ The VAE test. At one step no need for any detaches. """
    proposal,   target     = Gaussian1d(loc=1, name="q"), Gaussian1d(loc=4, name="p")
    fwd_kernel, rev_kernel = SimpleKernel(4, "fwd"),      SimpleKernel(4, "rev")

    # optimizer = torch.optim.Adam(dict(
    #     proposal=filter(lambda x: isinstance(x, Tensor), proposal.parameters()),
    #     fwd_kernel=filter(lambda x: isinstance(x, Tensor), fwd_kernel.parameters()),
    #     rev_kernel=filter(lambda x: isinstance(x, Tensor), rev_kernel.parameters()),
    #     target=filter(lambda x: isinstance(x, Tensor), target.parameters()),
    # ))
    optimizer = torch.optim.Adam(
        list(proposal.parameters()) + \
        list(fwd_kernel.parameters()) + \
        list(rev_kernel.parameters()) + \
        list(target.parameters())
    )

    num_steps = 1000
    with trange(num_steps) as bar:
        for i in bar:
            #=====================================
            # q_prp_trace, _ = proposal()
            # FIXME: for now this is acceptable, but long-term I think thereis a question of
            # 1. inspecting arguments of the proposal and fwd_kernel then compacting the call
            # 2. being able to pull out the trace of the proposal / kernel from forward?
            #   - right now how do you compute logprobs? you need to do this under the trace only, but it would
            #     be nice to just tuck `log_inc_weights` somewhere (maybe on Trace?).
            forward = Forward(fwd_kernel, proposal)
            q_ext_trace, _ = forward()
            # p_tar_trace, _ = target()
            p_mar_trace    = Reverse(target, rev_kernel)()

            log_weights    = p_mar_trace.log_joint(batch_dim=None) - q_ext_trace.log_joint(batch_dim=None)

            # tr, log_weights = Propose(Forward(fwd_kernel, proposal), Reverse(target, rev_kernel))

            p = proposal
            #

            for r, f, t in zip(forwards, reverses, targets):
                p = Propose(Reverse(t, r), Forward(f, p))
                out, ptrace, plog_probs, lweight, loss = p()
                #    ----------------------------------------------- State

            #=====================================
            loss = - log_weights.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 50 == 0:
                loss_scalar = log_weights.detach().cpu().sum().item()
                loss_template = 'loss={}{:.4f}'.format('' if loss_scalar < 0 else ' ', loss_scalar)
                bar.set_postfix_str(loss_template)

    with torch.no_grad():
        num_validate_samples = 1000
        samples = []
        for _ in range(validate_samples):
            q_ext_trace, out = Forward(fwd_kernel, proposal)()



@mark.skip()
def test_Gaussian_4step():
    """
    3-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).

    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
    """
    [prior, target1, target2, final] = [Gaussian1d(i) for i in range(4)]
