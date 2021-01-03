import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn as nn
from combinators import Program, Kernel, Forward, Reverse, Propose, Trace
from combinators.stochastic import RandomVariable, Provenance
from probtorch.util import expand_inputs
from typeguard import typechecked
from tqdm import trange
from pytest import mark
import sparklines
import combinators.trace.utils as trace_utils
from combinators.tensor.utils import thash, show
from typing import Optional


class MultivariateNormalKernel(nn.Module):
    def __init__(self, name, dim, dim_in=None, loc=None, cov=None, cov_embedding=None, map=None, reparameterized=True):
        super().__init__()
        self.dim = dim
        self.name = name
        self.optimizable = True

        if reparameterized is None:
            reparameterized = dist_type.has_rsample
        else:
            reparameterized = reparameterized
        super().__init__(name, dim)
        self.reparameterized = reparameterized
        self.name = name
        self.dim = dim
        self.dist_type = dist_type

        self.dim_in = dim if dim_in is None else dim_in
        self.cov_embedding = CovarianceEmbedding.SoftPlusDiagonal if cov_embedding is None else cov_embedding
        loc = torch.zeros(dim) if loc is None else loc
        cov = torch.eye(dim) if cov is None else cov
        cov_embed = self.cov_embedding.embed(cov, dim)
        super().__init__(name, dim, torch.distributions.MultivariateNormal, reparameterized=reparameterized)

        self.map = map
        if self.map is None:
            self.map = nets.ResMLPJ(dim_in=dim_in,
                                    dim_hidden=32,
                                    dim_out=dim),
        # initialize
        self.map.initialize(loc, cov_embed)

    def get_params(self, cond_set={}, param_set={}, detach_parameters=False):
        param_dict = self.parameter_map(cond_set, param_set)
        if detach_parameters:
            for name, param in param_dict.items():
                param_dict[name] = param.detach()
        return param_dict

    def get_log_density(self, cond_set={}, param_set={}, detach_parameters=False):
        return self.get_dist(cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters).log_prob

    def get_dist(self, cond_set={}, param_set={}, detach_parameters=False):
        return self.dist_type(**self.get_params(cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters))

    def forward(self, value=None, cond_set={}, param_set={}, detach_parameters=False, **kwargs):
        dist = self.get_dist(cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters)
        if value is None:
            if self.reparameterized:
                value = dist.rsample(**kwargs)
            else:
                value = dist.sample(**kwargs)
            provenance = Provenance.SAMPLED
        else:
            value = to_tensor(value)
            provenance = Provenance.OBSERVED
        return probtorch.RandomVariable(dist, value, provenance=provenance)

    def parameter_map(self, cond_set, param_set):
        out = self.map(to_tensor(cond_set['z_in']))
        loc, cov_embed = torch.split(out, [self.dim, self.cov_embedding.embed_dim(self.dim)], dim=-1)
        covariance_matrix = self.cov_embedding.unembed(cov_embed, self.dim)
        return {'loc': loc, 'covariance_matrix': covariance_matrix}

    def set_optimizable(self, b, recurse=True):
        if recurse:
            self.apply(lambda module: setattr(module, 'optimizable', b))
            self.requires_grad_(b)  # already recurive, i.e. sets requites_grad_(b) for all submodules!
        else:
            self.optimizable = b
            for param in self.parameters(recurse=False):
                param.requires_grad_(b)
        return self


class Net(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        # self.joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out))
        # self.sigma = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out), nn.ReLU())
        self.dim_out = dim_out

    def forward(self, x):
        # y = self.joint(x)
        mu = self.mu(x)
        # sigma = self.sigma(y)
        return mu, torch.ones([self.dim_out]) # *0.5 # sigma

class Gaussian1d(Program):
    def __init__(self, loc:int, name:str):
        super().__init__()
        self.loc = loc
        self.name = name
        self.size = torch.Size([1])

    def model(self, trace):
        return trace.normal(loc=torch.ones(self.size) *self.loc, scale=torch.ones(self.size), name=self.name)

class _Gaussian1d(nn.Module):
    def __init__(self, loc:int, std:int, name:str, num_samples:int):
        super().__init__()
        self.loc = loc
        self.std = std
        self.name = name
        self.size = torch.Size([1])
        self.expand_samples = lambda ten: ten.expand(num_samples, *ten.size())

    def forward(self, trace_name, cond_trace=None, num_samples=1):
        trace = Trace()
        out = trace.normal(
            loc=self.expand_samples(torch.ones(self.size, requires_grad=True)*self.loc),
            scale=self.expand_samples(torch.ones(self.size)*self.std),
            value=None if cond_trace is None else cond_trace[trace_name].value,

            name=trace_name)
        return out, trace

    def __repr__(self):
        return "[{}]{}".format(self.name, super().__repr__())


@typechecked
class _SimpleKernel(nn.Module):
    def __init__(self, num_hidden, name, ix=None):
        super().__init__()
        self.net = Net(dim_in=1, dim_hidden=num_hidden, dim_out=1)
        self.name = name if ix is None else name + "_" + str(ix)

    def forward(self, base_trace, base_name, trace_name, cond_trace=None, detach_base=False):
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
    # @expand_inputs
    def xforward(self, base_trace, base_name, trace_name, cond_trace=None):
        trace = Trace()
        # mu, sigma = out[0], out[1]
        base_rv = base_trace[base_name]
        copy_rv = RandomVariable(base_rv.dist, base_rv.value.detach(), provenance=Provenance.OBSERVED)
        trace.append(copy_rv, name=base_name)

        mu, sig = self.net(base_rv.value.detach())
        # import ipdb; ipdb.set_trace();

        ext_dist = Normal(loc=mu, scale=sig)
        ext_rv = RandomVariable(ext_dist, (ext_dist.sample() if cond_trace is None else cond_trace[trace_name].value), provenance=Provenance.SAMPLED)
        trace.append(ext_rv, name=trace_name)

        return ext_rv.value, trace
    def __repr__(self):
        return "[{}]{}".format(self.name, super().__repr__())

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
    return myestimate_mc(lvs, lws.detach())

def _nvo_avo(
    lw: Tensor,  # Log cumulative weight, this is not detached yet - see if it bites Babak and Hao
    lv: Tensor,
    proposal_trace: Trace,
    target_trace: Trace,
    *args,
    batch_dim=1,
    sample_dims=0,
    reducedims=None,
    **kwargs,
) -> Tensor:
    """
    IMPORTANT: Only applicable for balanced NVI-Sequential!
    Metric: Exp[log(g_k(z_k) r_k(z_k-1 | z_k)) - log(q_k-1(z_k-1) q_k(z_k | z_k-1))]
    Notes: -> detach q_k
    """
    if reducedims is None:
        reducedims = (sample_dims,)

    lw = lw.detach()
    loss = _estimate_mc(-lv, torch.zeros_like(lv),
                        sample_dims=sample_dims,
                        reducedims=reducedims,
                        keepdims=False,
                        )
    return loss

def _estimate_mc(
    values: Tensor,
    log_weights: Tensor,
    sample_dims: torch.Size,
    reducedims: torch.Size,
    keepdims: bool,
) -> Tensor:
    nw = torch.nn.functional.softmax(log_weights, dim=sample_dims)
    return (nw * values).sum(dim=reducedims, keepdim=keepdims)

def myestimate_mc(values: Tensor, log_weights: Tensor) -> Tensor:
    return _estimate_mc(values, log_weights, sample_dims=0, reducedims=0, keepdims=False)

def properly_weight(pw_op, target_trace):
    """
    Applies a proper weighting operation to the target trace.
    Applying a proper weighting operation results in a new log weight and a - possibly modified - target trace which act
    as a properly weighted pair for the desity defined by the target trace.
    """
    return pw_op(target_trace)

def trace(stochastic_gen, name, value=None, cond_map={}, param_set={}, detach_value=False, **kwargs):
    return extend(stochastic_gen, name, Trace(), value=value,
                       cond_map=cond_map, param_set=param_set,
                       detach_value=detach_value, **kwargs)

def trace_with(stochastic_gen, name, cond_trace, value=None, cond_map={}, param_set={}, detach_value=False, **kwargs):
    return extend_with(stochastic_gen, name, Trace(), cond_trace, value=value,
                       cond_map=cond_map, param_set=param_set,
                       detach_value=detach_value, **kwargs)

def to_tensor(val, detach=False):
    val = val.value
    if detach:
        val = val.detach()
    return val

def extend(stochastic_gen, name, trace, value=None, cond_map={}, param_set={},
           detach_value=False, detach_parameters=False, **kwargs):
    # If there is a cond. trace, check if new rv is observed, i.e. rv already exists in cond. trace
    if name in trace:
        value = trace[name]

    if value is not None:
        value = to_tensor(value, detach_value)

    # Build conditioning set from rvs in cond. trace or from trace
    cond_set = {}
    for var_name_gen, var_name in cond_map.items():
        cond_set[var_name_gen] = to_tensor(trace[var_name], detach_value)

    # Get stochastic nodes and append them to the trace
    node = stochastic_gen.forward(value=value, cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters, **kwargs)
    node.generator = stochastic_gen
    trace.append(node, name=name)
    assert not (detach_value and not (node.value.grad_fn is None))
    return trace

def extend_with(stochastic_gen, name, trace, cond_trace, value=None, cond_map={}, param_set={},
           detach_value=False, detach_parameters=False, **kwargs):
    # If there is a cond. trace, check if new rv is observed, i.e. rv already exists in cond. trace
    if name in cond_trace:
        value = cond_trace[name]

    if value is not None:
        value = to_tensor(value, detach_value)

    # Build conditioning set from rvs in cond. trace or from trace
    cond_set = {}
    for var_name_gen, var_name in cond_map.items():
        cond_set[var_name_gen] = to_tensor(cond_trace[var_name], detach_value)

    # Get stochastic nodes and append them to the trace
    node = stochastic_gen.forward(value=value, cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters, **kwargs)
    node.generator = stochastic_gen
    trace.append(node, name=name)
    assert not (detach_value and not (node.value.grad_fn is None))
    return trace

def xtend(stochastic_gen, name, trace, cond_trace=None, value=None, cond_map={}, param_set={},
           detach_value=False, detach_parameters=False, **kwargs):
    if cond_trace is None:
        cond_trace = trace
    # If there is a cond. trace, check if new rv is observed, i.e. rv already exists in cond. trace
    if name in cond_trace:
        value = cond_trace[name]

    if value is not None:
        value = to_tensor(value, detach_value)

    # Build conditioning set from rvs in cond. trace or from trace
    cond_set = {}
    for var_name_gen, var_name in cond_map.items():
        cond_set[var_name_gen] = to_tensor(cond_trace[var_name], detach_value)

    # Get stochastic nodes and append them to the trace
    node = stochastic_gen.forward(value=value, cond_set=cond_set, param_set=param_set, detach_parameters=detach_parameters, **kwargs)
    node.generator = stochastic_gen
    trace.append(node, name=name)
    assert not (detach_value and not (node.value.grad_fn is None))
    return trace

def step(k, p_prev, proposal, target, fwd_kernel, rev_kernel, num_steps):
    # Construct extended proposal
    # Don't detach in first step
    # q_prop = proposal.trace(z(k), cond_trace=p_prev, detach_value=False if k == 0 else True)
    q_prop = trace_with(proposal, "z_{:d}".format(k), cond_trace=p_prev, detach_value=True)
    q_ext = trace(fwd_kernel, "z_{:d}".format(k+1), q_prop, cond_map={'z_in': "z_{:d}".format(k)})

    # Construct extended target
    p_tar = trace_with(target, "z_{:d}".format(k+1), cond_trace=q_ext)
    p_ext = extend_with(rev_kernel, "z_{:d}".format(k), p_tar, cond_trace=q_ext, cond_map={'z_in': "z_{:d}".format(k+1)})

    # IW relabeling:
    lv = trace_utils.log_importance_weight(proposal_trace=q_ext, target_trace=p_ext, batch_dim=1, sample_dims=0, check_exist=True)
    p_prop = p_ext
    lv_prop = lv
    return q_ext, p_ext, lv, p_prop, lv_prop


def test_Gaussian_sanitycheck():
    torch.manual_seed(1)
    target_mean, target_stdv = 4, 1
    num_samples=200
    lr = 1e-2
    proposal = _Gaussian1d(loc=target_stdv, std=6, name='g_0', num_samples=num_samples)
    target = _Gaussian1d(loc=target_mean, std=1, name='g_1', num_samples=num_samples)
    fwd = _SimpleKernel(num_hidden=4, name='f_01')
    rev = _SimpleKernel(num_hidden=4, name='r_10')
    optimizer = torch.optim.Adam([dict(params=x.parameters()) for x in [proposal, target, fwd, rev]], lr=lr)

    num_steps = 150
    loss_ct, loss_sum = 0, 0.0
    loss_all = []
    loss_every = []
    def print_traces(ext_only=True):
        if not ext_only:
            try:
                print("p_prv: {}".format(trace_utils.show(p_prv, fix_width=True)))
            except:
                print("p_prv: None")
            try:
                print("q_prp: {}".format(trace_utils.show(q_prp, fix_width=True)))
            except:
                print("q_prp: None")
        try:
            print("q_ext: {}".format(trace_utils.show(q_ext, fix_width=True)))
        except:
            print("q_ext: None")
        if not ext_only:
            try:
                print("p_tar: {}".format(trace_utils.show(p_tar, fix_width=True)))
            except:
                print("p_tar: None")
        try:
            print("p_ext: {}".format(trace_utils.show(p_ext, fix_width=True)))
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
            # TODO: reduce number of samples, and just fix the samples to something simple

            _, q_prp = proposal(  trace_name='z_0',                 cond_trace=p_prv) # q_prp: {z_0: normal(1,6).detach()}
            # NOTE: ALL OF p_prv is detached

            # TODO: Remove joint nn. Fix weights of NN to [xavier?]
            _, q_ext = fwd(q_prp, trace_name='z_1', base_name='z_0')                    # q_ext: {z_0: normal(1,6).detach(), z_1: normal(nnet(q_prp['z_0'].value.detach()),1).detach()}

            # NOTE: z_0 must _not_ have grad
            # NOTE: z_1 must have grad
            # NOTE: AVO ONLY WORKS FOR REPARAMETERIZED <<< maybe probtorch does this already. doublecheck
            #  - check if samples have grad

            # TODO: same as with proposal above
            _, p_tar = target(    trace_name='z_1',                   cond_trace=q_ext) # p_tar: {z_1: normal(4,1).detach()}
            # NOTE: z_1 must have grad

            # TODO: same as with fwd above
            _, p_ext = rev(p_tar, trace_name='z_0', base_name='z_1',  cond_trace=q_ext) # q_ext: {z_0: normal(nnet(p_tar['z_1'].value.detach()), 6).detach(), z_1: normal(4,1).detach()}
            # NOTE: z_1 must have grad <<< note!!! important for connecting the gradients
            # NOTE: z_0 must have grad
            # q_ext['z_0'].value.requires_grad = False
            # print_traces(ext_only=False)
            #import ipdb; ipdb.set_trace();

            q_ext = trace_utils.copytrace(q_ext, detach=p_prv.keys())
            # ================== #
            # Compute Weight     #
            # ================== #

            assert trace_utils.valeq(q_ext, p_ext, nodes=p_ext._nodes, check_exist=True)
            lv = p_ext.log_joint(batch_dim=None, sample_dims=0, nodes=p_ext._nodes) - \
                q_ext.log_joint(batch_dim=None, sample_dims=0, nodes=p_ext._nodes)
            # ^^^^ everything will work out with reparams
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

            # loss = loss_k
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_ct += 1
            loss_scalar = lv.detach().cpu().mean().item()
            loss_sum += loss_scalar
            loss_every.append(loss_scalar)
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
        tlosses = torch.tensor(loss_all)
        lowest_loss = min(loss_all)
        highest_loss = max(loss_all)
        print("greatest loss: {:.4f}\nsmallest loss: {:.4f}".format(lowest_loss, highest_loss))
        print(sparklines.sparklines(((-tlosses) + abs(tlosses.min())).numpy())[0])

        num_validate_samples = 400
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





@mark.skip()
def test_Gaussian_1step():
    """ The VAE test. At one step no need for any detaches. """
    target_mean, target_stdv = 4, 1

    proposal,   target     = Gaussian1d(loc=1, name="q"), Gaussian1d(loc=target_mean, name="p")
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
    loss_ct, loss_sum = 0, 0.0
    loss_all = []
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
            reverse = Reverse(target, rev_kernel)
            extended = Propose(forward, reverse)
            tr, log_weights = extended()

            #

            #=====================================
            loss = - log_weights.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_ct += 1
            loss_scalar = log_weights.detach().cpu().sum().item()
            loss_sum += loss_scalar
            if i % 10 == 0:
                loss_avg = loss_sum / loss_ct
                loss_template = 'loss={}{:.4f}'.format('' if loss_avg < 0 else ' ', loss_avg)
                bar.set_postfix_str(loss_template)
                loss_ct, loss_sum  = 0, 0.0
                loss_all.append(loss_avg)

    import sparklines
    print("min: {:.4f}\nmax: {:.4f}".format(max(loss_all), min(loss_all)))
    print(sparklines.sparklines(list(map(lambda l: 0 if l < 0 else l, loss_all)))[0])
    print(sparklines.sparklines(loss_all)[0])

    with torch.no_grad():
        num_validate_samples = 1000
        samples = []
        for _ in range(num_validate_samples):
            q_ext_trace, out = Forward(fwd_kernel, proposal)()
            samples.append(out.cpu().item())
        evaluation = torch.tensor(samples)
        eval_mean, eval_stdv = evaluation.mean().item(), evaluation.std().item()
        mean_tol, stdv_tol = 0.25, 0.5
        assert (target_mean - mean_tol) < eval_mean and  eval_mean < (target_mean + mean_tol)
        assert (target_stdv - stdv_tol) < eval_stdv and  eval_stdv < (target_stdv + stdv_tol)



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
