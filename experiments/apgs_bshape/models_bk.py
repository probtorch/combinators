from collections import namedtuple
from combinators.inference import IX
from combinators.utils import ppr
from combinators.kernel import Kernel
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import math
import torch.nn.functional as F
import combinators.trace.utils as trace_utils
import combinators.tensor.utils as tensor_utils
from combinators import Program, debug

def copy_trace(trace, exclude_name):
    return trace_utils.copysubtrace(trace, set(trace.keys()) - {exclude_name})

sweepix = namedtuple('sweepix', ['sweep', 'rev', 'block', 't', 'recon_level'])
ix = sweepix

getmap = lambda name: (lambda ix: (f'{name}{ix.sweep if ix.rev else ix.sweep-1}', f'{name}{ix.sweep-1 if ix.rev else ix.sweep}'))

# Encoder for z_where - Hao's version
class Enc_coor(nn.Module):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, reparameterized=False):
        super().__init__()
        self.enc_coor_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim))
        self.reparameterized = reparameterized
#         self.conv_kernel = conv_kernel
        self.AT = AT

    def forward(self, q, frames, timestep, conv_kernel, extend_dir):
        # if timestep <= IX:
        #     ppr(q, desc=f"Enc_coor inp {timestep}->{timestep+1}")
        debug.seed(timestep);
        _, _, K, DP, _ = conv_kernel.shape
        S, B, T, FP, _ = frames.shape
        frame_left = frames[:,:,timestep,:,:]
        q_mean = []
        q_std = []
        z_where_t = []
        for k in range(K):
            conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), conv_kernel[:,:,k,:,:].view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            hidden = self.enc_coor_hidden(conved_k)
            q_mean_k = self.where_mean(hidden)
            q_std_k = self.where_log_std(hidden).exp()
            q_mean.append(q_mean_k.unsqueeze(2))
            q_std.append(q_std_k.unsqueeze(2))
            if extend_dir == 'forward':
                if self.reparameterized:
                    z_where_k = Normal(q_mean_k, q_std_k).rsample()
                else:
                    z_where_k = Normal(q_mean_k, q_std_k).sample()
                z_where_t.append(z_where_k.unsqueeze(2))
            elif extend_dir == 'backward':
                z_where_k = q['z_where_%d' % (timestep+1)].value[:,:,k,:]
            recon_k = self.AT.digit_to_frame(conv_kernel[:,:,k,:,:].unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            assert recon_k.shape ==(S,B,FP,FP), 'shape = %s' % recon_k.shape
            frame_left = frame_left - recon_k

        q_mean = torch.cat(q_mean, 2)
        # if ('z_where_%d' % (timestep+1)) in q and torch.allclose(q_mean, q['z_where_%d' % (timestep+1)].value):
        #     breakpoint()
        q_std = torch.cat(q_std, 2)

        q_new = copy_trace(q, 'z_where_%d' % (timestep+1))
        if extend_dir == 'forward':
            z_where_t = torch.cat(z_where_t, 2)
            z_where_tp1 = q_new.normal(loc=q_mean, scale=q_std, value=z_where_t, name='z_where_%d' % (timestep+1))
        elif extend_dir == 'backward':
            try:
                z_where_old = q['z_where_%d' % (timestep+1)].value
            except:
                print("cannot extract z_where_%d from the incoming trace." % (timestep+1))
            q_new.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where_%d' % (timestep+1))
        else:
            raise ValueError
        #print(tensor_utils.show(z_where_t), tensor_utils.show(z_where_tp1))
        # if timestep <= IX:
        #     ppr(q_new, desc=f"Enc_coor out {timestep}->{timestep+1}")
        return q_new

# Encoder for z_where in (old) Combinators
class Enc_coor2(Kernel):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, reparameterized=False):
        super().__init__()
        self.enc_coor_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim))
        self.reparameterized = reparameterized
#         self.conv_kernel = conv_kernel
        self.AT = AT

    # apply_kernel --rename-> model
    # cond_output --rename-> c
    # cond_trace is not there anymore
    def apply_kernel(self, trace, cond_trace, cond_output, ix=None):
        debug.seed(ix.t);
        try:
            frames = cond_output['frames']
        except:
            raise Exception(ix)

        if 'conv_kernel' not in cond_output or cond_output['conv_kernel'] is None:
            conv_kernel = cond_output['get_conv'](eval_output=cond_output, ix=ix)
            cond_output['conv_kernel'] = conv_kernel
        else:
            conv_kernel = cond_output['conv_kernel']

        timestep = ix.t

        # =============================================================== #
        _, _, K, DP, _ = conv_kernel.shape
        S, B, T, FP, _ = frames.shape
        frame_left = frames[:,:,timestep,:,:]
        q_mean = []
        q_std = []
        z_where_t = []
        for k in range(K):
            conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), conv_kernel[:,:,k,:,:].view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            hidden = self.enc_coor_hidden(conved_k)
            q_mean_k = self.where_mean(hidden)
            q_std_k = self.where_log_std(hidden).exp()
            q_mean.append(q_mean_k.unsqueeze(2))
            q_std.append(q_std_k.unsqueeze(2))
            if not ix.rev:
                dist = Normal(q_mean_k, q_std_k)
                z_where_k = dist.rsample() if self.reparameterized else dist.sample()
                z_where_t.append(z_where_k.unsqueeze(2))
            else:
                z_where_k = cond_output['z_where_%d' % (timestep+1)][:,:,k,:]

            recon_k = self.AT.digit_to_frame(conv_kernel[:,:,k,:,:].unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            assert recon_k.shape ==(S,B,FP,FP), 'shape = %s' % recon_k.shape
            frame_left = frame_left - recon_k
        q_mean = torch.cat(q_mean, 2)
        q_std = torch.cat(q_std, 2)

        if not ix.rev:
            z_where_t = torch.cat(z_where_t, 2)
            z_where_tp1 = trace.normal(loc=q_mean, scale=q_std, value=z_where_t, name='z_where_%d' % (timestep+1))
            cond_output['z_where_T'].append(z_where_t.unsqueeze(2))
        else:
            try:
                z_where_old = cond_output['z_where_%d' % (timestep+1)]
            except:
                print("cannot extract z_where_%d from the incoming trace." % (timestep+1))
            z_where_tp1 = trace.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where_%d' % (timestep+1))

        output = cond_output
        output[f'z_where_{ix.t+1}'] = z_where_tp1
        return output

# This can be deleted
class Noop(Program):
    def __init__(self):
        super().__init__()

    def model(self, _, *args, ix=None):
        if len(args) == 0:
            return dict()
        elif len(args) == 1 and isinstance(args[0], dict):
            return args[0]
        else:
            raise RuntimeError()

# This can be deleted as well
class Memo(Kernel):
    """ turns a program into a kernel """
    def __init__(self, program):
        super().__init__()
        self.program = program

    def apply_kernel(self, trace, *args, **kwargs):
        out = self.program(*args, **kwargs)
        trace.update(out.trace)
        return out.output

class Dec_coor(nn.Module):
    """
    generative model of digit positions
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0) : S * B * D
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, z_where_dim):
        super().__init__()
        self.prior_mu0 = torch.zeros(z_where_dim)
        self.prior_Sigma0 = torch.ones(z_where_dim) * 1.0
        self.prior_Sigmat = torch.ones(z_where_dim) * 0.2

    def forward(self, q, p, timestep, recon_level='obj'):
        debug.seed(timestep);
        # if timestep <= IX:
        #     print("<><><><><><><><><><>")
        #     ppr(p, desc=f"Dec_coor inp {timestep}->{timestep+1}")
        #     
        if timestep == 0:
            p.normal(loc=self.prior_mu0,
                     scale=self.prior_Sigma0,
                     value=q['z_where_%d' % (timestep+1)].value,
                     name='z_where_%d' % (timestep+1))
        else:
            p.normal(loc=q['z_where_%d' % (timestep)].value,
                     scale=self.prior_Sigmat,
                     value=q['z_where_%d' % (timestep+1)].value,
                     name='z_where_%d' % (timestep+1))
        # if timestep <= IX:
        #     ppr(p, desc=f"Dec_coor out {timestep}->{timestep+1}")
        #     print("++++++++++++++++++++")
        return p

class Dec_coor2(Program):
    """
    generative model of digit positions
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0) : S * B * D
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, z_where_dim):
        super().__init__()
        self.prior_mu0 = torch.zeros(z_where_dim)
        self.prior_Sigma0 = torch.ones(z_where_dim) * 1.0
        self.prior_Sigmat = torch.ones(z_where_dim) * 0.2

    def model(self, trace, eval_output, shared_args, ix=None):
        z_where_str = lambda t: f'z_where_{t}'
        get_z_where = lambda t: eval_output[z_where_str(t)]
        trace.normal(loc=self.prior_mu0, scale=self.prior_Sigma0, value=get_z_where(1), name=z_where_str(1))
        for t in range(1, ix.t-1 if ix.block == "is" else ix.t+1):
            trace.normal(loc=get_z_where(t), scale=self.prior_Sigmat, value=get_z_where(t+1), name=z_where_str(t+1))
        return eval_output


class Enc_digit(nn.Module):
    """
    encoder of digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT, reparameterized=False):
        super(self.__class__, self).__init__()
        self.enc_digit_hidden = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, int(0.5*num_hidden)),
                        nn.ReLU())
        self.enc_digit_mean = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.enc_digit_log_std = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))

        self.reparameterized = reparameterized
        self.AT = AT

    def forward(self, q, frames, extend_dir, timestep):
        debug.seed(timestep)
        z_where = []
        for t in range(frames.shape[2]):
            z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
        z_where = torch.cat(z_where, 2)
        # print(tensor_utils.show(z_where))
        cropped = self.AT.frame_to_digit(frames=frames, z_where=z_where)
        cropped = torch.flatten(cropped, -2, -1)
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden)
        q_std = self.enc_digit_log_std(hidden).exp()
        q_new = copy_trace(q, 'z_what')
        if extend_dir == 'forward':
            if self.reparameterized:
                z_what = Normal(q_mu, q_std).rsample()
            else:
                z_what = Normal(q_mu, q_std).sample() ## S * B * K * z_what_dim
            q_new.normal(loc=q_mu, scale=q_std, value=z_what, name='z_what')
        elif extend_dir == 'backward':
            try:
                z_what_old = q['z_what'].value
            except:
                print('cannot extract z_what from the incoming trace')
            q_new.normal(loc=q_mu, scale=q_std, value=z_what_old, name='z_what')
        else:
            raise ValueError
        return q_new


class Enc_digit2(Kernel):
    """
    encoder of digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT, reparameterized=False):
        super().__init__()
        self.enc_digit_hidden = nn.Sequential(nn.Linear(num_pixels, num_hidden), nn.ReLU(),
                                              nn.Linear(num_hidden, int(0.5*num_hidden)), nn.ReLU())
        self.enc_digit_mean = nn.Sequential(nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.enc_digit_log_std = nn.Sequential(nn.Linear(int(0.5*num_hidden), z_what_dim))

        self.reparameterized = reparameterized
        self.AT = AT

    def apply_kernel(self, q_new, cond_trace, cond_output, ix=None):
        assert ix is not None
        # print(f'[ℇ] digit2 {ix}')
        debug.seed(ix.t);
        q = cond_trace
        frames = cond_output['frames']

        if ix.block == 'is' or (ix.block is "what" and not ix.rev):
            cond_output['z_where'] = z_where = torch.cat(cond_output['z_where_T'], 2)
            cond_output['z_where_T'] = []
        else:
            z_where = cond_output['z_where']

        cropped = self.AT.frame_to_digit(frames=frames, z_where=z_where)
        cropped = torch.flatten(cropped, -2, -1)
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden)
        q_std = self.enc_digit_log_std(hidden).exp()
        if not ix.rev:
            if self.reparameterized:
                z_what = Normal(q_mu, q_std).rsample()
            else:
                z_what = Normal(q_mu, q_std).sample() ## S * B * K * z_what_dim
            zwhat = q_new.normal(loc=q_mu, scale=q_std, value=z_what, name='z_what')
            cond_output['z_what'] = zwhat
        else:
            try:
                z_what_old = cond_output['z_what']
            except:
                print('cannot extract z_what from the incoming trace')
                raise
            zwhat = q_new.normal(loc=q_mu, scale=q_std, value=z_what_old, name='z_what')
            cond_output['z_what'] = zwhat
        return cond_output




class Dec_digit(nn.Module):
    """
    decoder of the digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT):
        super(self.__class__, self).__init__()
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        self.AT = AT

    def forward(self, q, p, frames, recon_level, timestep=None):
        debug.seed(timestep);
        digit_mean = self.dec_digit_mean(q['z_what'].value)  # S * B * K * (28*28)

        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        if recon_level == 'object': ## return the recnostruction of objects
            return digit_mean.detach()

        elif recon_level == 'frame':
            _, _, FP, _ = frames.shape
            z_where = q['z_where_%d' % (timestep+1)].value.unsqueeze(2)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0).squeeze() # S * B * T * FP * FP
            assert recon_frames.shape == frames.shape, 'recon_frames shape =%s, frames shape = %s' % (recon_frames.shape, frames.shape)
            _ = p.variable(Bernoulli, probs=recon_frames, value=frames, name='recon')
            return p

        elif recon_level =='frames': # return the reconstruction of the entire frames
            _, _, T, FP, _ = frames.shape
            z_where = []
            for t in range(T):
                z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
            z_where = torch.cat(z_where, 2)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
            assert recon_frames.shape == (S, B, T, FP, FP), "ERROR! unexpected reconstruction shape"
            p.normal(loc=self.prior_mu,
                     scale=self.prior_std,
                     value=q['z_what'].value,
                     name='z_what')
            _= p.variable(Bernoulli, probs=recon_frames, value=frames, name='recon')
            return p
        else:
            raise ValueError


class Dec_digit2(Program):
    """
    decoder of the digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT):
        super().__init__()
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        self.AT = AT

    def model(self, trace, eval_output, shared_args, ix=None):
        # WARNING: slight deviation in IS of the recon RV
        assert ix is not None
        timestep=ix.t
        debug.seed(timestep);
        if ix.block == "is":
            # we are in the kernel position of a forward:
            # eval_output == cond_trace
            # shared_args == cond_output
            frames = shared_args['frames']
            digit_mean = self.dec_digit_mean(shared_args['z_what'])  # S * B * K * (28*28)

            self.add_likelihood_to_trace(
                trace, frames,
                z_what=shared_args['z_what'], z_where=shared_args['z_where'],
                digit_mean=digit_mean)
            return shared_args
        else:
            frames = eval_output['frames']
            try:
                digit_mean = self.dec_digit_mean(eval_output['z_what'])  # S * B * K * (28*28)
            except:
                breakpoint();
                raise Exception(ix)

            S, B, K, DP2 = digit_mean.shape
            DP = int(math.sqrt(DP2))
            digit_mean = digit_mean.view(S, B, K, DP, DP)
            # EValuation
            if ix.recon_level == 'object': ## return the recnostruction of objects
                # print(f'[ⅆ] eval {ix}')
                eval_output['conv_kernel'] = digit_mean.detach()
                return eval_output

            # EValuation
            # FIXME:
            elif ix.recon_level == 'frame':
                # reconstruct_image(self, trace, frames, recon_frames, z_what, z_where_T, digit_mean, sample=False):
                pass

            # likeihood
            elif ix.recon_level =='frames': # return the reconstruction of the entire frames
                # print(f'[ⅆ] digit2 {ix}')
                self.add_likelihood_to_trace(self, trace, frames, z_what=eval_output['z_what'], z_where=eval_output['z_where'], digit_mean=digit_mean)
                return eval_output
            else:
                raise ValueError

    def add_likelihood_to_trace(self, trace, frames, z_what, z_where, digit_mean):
        S, B, K, DP2 = digit_mean.shape
        _, _, T, FP, _ = frames.shape
        recon_frames = torch.clamp(self.AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
        assert recon_frames.shape == (S, B, T, FP, FP), "ERROR! unexpected reconstruction shape"
        trace.normal(loc=self.prior_mu,
                 scale=self.prior_std,
                 value=z_what,
                 # should relabel this to *_ll
                 name='z_what')
        _= trace.variable(Bernoulli, probs=recon_frames, value=frames, name='recon')

    def reconstruct_image(self, trace, frames, z_what, z_where_T, digit_mean, sample=False):
        rs = []
        for _z_where in z_where_T:
            z_where = z_where.unsqueeze(2)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0).squeeze() # S * B * T * FP * FP
            rs.append(recon_frames)
            assert recon_frames.shape == frames.shape, 'recon_frames shape =%s, frames shape = %s' % (recon_frames.shape, frames.shape)
            _ = trace.variable(Bernoulli, probs=recon_frames, value=frames, name='recon_frame')
        return rs
