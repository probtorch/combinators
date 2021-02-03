# -*- whitespace-line-column: 100; -*-
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import combinators.stochastic as probtorch
import math
from combinators.tensor.utils import autodevice
from combinators.utils import load_models, adam
from combinators import Program
from combinators.trace.utils import copy_trace

sweepix = namedtuple('sweepix', ['sweep', 'rev', 'block', 't'])
ix = sweepix

getmap = lambda name: (lambda ix: (f'{name}{ix.sweep if ix.rev else ix.sweep-1}', f'{name}{ix.sweep-1 if ix.rev else ix.sweep}'))
# pmap = getmap('precisions')
# mmap = getmap('means')
# smap = getmap('states')

def init_models(AT=AT, frame_pixels, digit_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, device,
    load=False,
    load_folder=None,
    load_filename=None,
):
    models = dict(
        enc_coor = Enc_coor(AT=AT, num_pixels=(frame_pixels-digit_pixels+1)**2, num_hidden=num_hidden_coor, z_where_dim=z_where_dim).to(autodevice(device)),
        dec_coor = Dec_coor(AT=AT, z_where_dim=z_where_dim).to(autodevice(device)),
        enc_digit = Enc_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT).to(autodevice(device)),
        dec_digit = Dec_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT).to(autodevice(device)),
    )
    return load_models(models, filename=load_filename, weights_dir=load_folder) if load else models

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
        self.AT = AT

    def forward(self, q, frames, timestep, conv_kernel, extend_dir):
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
        q_std = torch.cat(q_std, 2)
        q_new = copy_trace(q, 'z_where_%d' % (timestep+1))
        if extend_dir == 'forward':
            z_where_t = torch.cat(z_where_t, 2)
            q_new.normal(loc=q_mean, scale=q_std, value=z_where_t, name='z_where_%d' % (timestep+1))
        elif extend_dir == 'backward':
            try:
                z_where_old = q['z_where_%d' % (timestep+1)].value
            except:
                print("cannot extract z_where_%d from the incoming trace." % (timestep+1))
            q_new.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where_%d' % (timestep+1))
        else:
            raise ValueError
        return q_new

class Enc_coor(Program):
    """ encoder of the digit positions """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, reparameterized=False):
        super().__init__()
        self.reparameterized = reparameterized
        self.enc_coor_hidden = nn.Sequential(nn.Linear(num_pixels, num_hidden), nn.ReLU())
        self.AT = AT

        self.where_mean = \
            nn.Sequential(
                nn.Linear(num_hidden, int(0.5*num_hidden)), nn.ReLU(),
                nn.Linear(int(0.5*num_hidden), z_where_dim), nn.Tanh())

        self.where_log_std = \
            nn.Sequential(
                nn.Linear(num_hidden, int(0.5*num_hidden)), nn.ReLU(),
                nn.Linear(int(0.5*num_hidden), z_where_dim))

    def haoimpl(self, q, conved, sampled=True, z_where_old=None, patch=False, template=None, k=None, frame=None, frame_left=None, AT=None):
        if patch:
            FP = frame.shape[-1]
            S, B, K, DP, _ = template.shape
            template_k = template[:,:,k,:,:]
            conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), template_k.view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            conved = conved_k

        hidden = self.enc_coor_hidden(conved)
        q_mean = self.where_mean(hidden)
        q_std = self.where_log_std(hidden).exp()
        if sampled:
            if self.reparameterized:
                z_where = Normal(q_mean, q_std).rsample()
            else:
                z_where = Normal(q_mean, q_std).sample()
            out = q.normal(loc=q_mean, scale=q_std, value=z_where, name='z_where')
        else:
            out = q.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where')

        return template_k, out

    def model(self, q, conved, sampled=True, z_where_old=None, ix=None, patch=False, template=None, k=None, frame=None, frame_left=None, AT=None):
        """Enc_coor"""
        if ix is None:
            return self.haoimpl(q, conved, sampled=sampled, z_where_old=z_where_old, patch=patch, template=template, k=k, frame=frame, frame_left=frame_left, AT=AT)

        hidden = self.enc_coor_hidden(conved)
        q_mean = self.where_mean(hidden)
        q_std  = self.where_log_std(hidden).exp()
        if sampled:
            z_where_dist = Normal(q_mean, q_std)
            value = z_where_dist.rsample() if self.reparameterized else z_where_dist.sample()
        else:
            value=z_where_old
        q.normal(loc=q_mean, scale=q_std, value=value, name='z_where')
        return q

class Enc_digit(Program):
    """ encoder of digit features """
    def __init__(self, num_pixels, num_hidden, z_what_dim, reparameterized=False):
        super().__init__()
        self.reparameterized = reparameterized
        self.enc_digit_hidden = \
            nn.Sequential(
                nn.Linear(num_pixels, num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, int(0.5*num_hidden)), nn.ReLU())

        self.enc_digit_mean    = nn.Sequential(nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.enc_digit_log_std = nn.Sequential(nn.Linear(int(0.5*num_hidden), z_what_dim))

    def haoimpl(self, q, cropped, sampled=True, z_what_old=None):
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden) ## because T is on the 3rd dim in cropped
        q_std = self.enc_digit_log_std(hidden).exp()
        if sampled:
            z_what_dist = Normal(q_mu, q_std)   ## S * B * K * z_what_dim
            value = z_what_dist.rsample() if self.reparameterized else z_what_dist.sample()
        else:
            value=z_what_old,
        q.normal(loc=q_mu, scale=q_std, value=value, name='z_what')
        return q

    def model(self, q, cropped, sampled=True, z_what_old=None, ix=None):
        """ Enc_digit """
        if ix is None:
            return self.haoimpl(q, cropped, sampled=sampled, z_what_old=z_what_old)

        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden) ## because T is on the 3rd dim in cropped
        q_std = self.enc_digit_log_std(hidden).exp()
        if sampled:
            z_what_dist = Normal(q_mu, q_std)   ## S * B * K * z_what_dim
            value = z_what_dist.rsample() if self.reparameterized else z_what_dist.sample()
        else:
            value=z_what_old,
        q.normal(loc=q_mu, scale=q_std, value=value, name='z_what')
        return None


class Dec_coor(Program):
    """
    generative model of digit positions
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0) : S * B * D
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, z_where_dim):
        super().__init__()
        self.prior_mu0    = torch.zeros(z_where_dim)
        self.prior_Sigma0 = torch.ones(z_where_dim) * 1.0
        self.prior_Sigmat = torch.ones(z_where_dim) * 0.2

    def haoimpl(self, z_where_t, z_where_t_1=None, patch=False, template=None, k=None, frame=None, frame_left=None, AT=None):
        if patch:
            template_k = template[:,:,k,:,:]
            recon_k = AT.digit_to_frame(template_k.unsqueeze(2), z_where_t.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2) ## S * B * 64 * 64

        S, B, D = z_where_t.shape
        if z_where_t_1 is None:
            p0 = Normal(self.prior_mu0, self.prior_Sigma0)
        else:
            p0 = Normal(z_where_t_1, self.prior_Sigmat)
        return dict(recon_k=recon_k,
                    p0=p0.log_prob(z_where_t).sum(-1), # S * B
                    frame_left = frame_left - recon_k
                    )

    def log_prior(self, z_where_t, z_where_t_1=None):
        """ seems like a useless function """
        self.haoimpl(z_where_t, z_where_t_1=z_where_t_1).sum(-1) # ?????

    def model(self, trace, z_where_t, z_where_t_1=None, ix=None, patch=False, template=None, k=None, frame=None, frame_left=None, AT=None):
        """ Dec_coor: S, B, D = z_where_t.shape """
        if ix is None:
            return self.haoimpl(z_where_t, z_where_t_1=z_where_t_1, patch=patch, AT=AT, template=template, k=k, frame=frame, frame_left=frame_left)

        if z_where_t_1 is None:
            p0 = Normal(self.prior_mu0, self.prior_Sigma0)
        else:
            p0 = Normal(z_where_t_1, self.prior_Sigmat)
        return p0.log_prob(z_where_t).sum(-1)# S * B


class Dec_digit(Program):
    """ decoder of the digit features """
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super().__init__()
        self.dec_digit_mean = \
            nn.Sequential(
                nn.Linear(z_what_dim, int(0.5*num_hidden)), nn.ReLU(),
                nn.Linear(int(0.5*num_hidden), num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, num_pixels), nn.Sigmoid())

        self.prior_mu  = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

    def model(self, trace, frames, z_what, z_where=None, AT=None):
        digit_mean = self.dec_digit_mean(z_what)  # S * B * K * (28*28)
        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        if z_where is None:
            ## return the recnostruction of mnist image
            return digit_mean.detach()
        else: # return the reconstruction of the frame
            assert AT is not None, "ERROR! NoneType variable AT found."
            assert frames is not None, "ERROR! NoneType variable frames found."
            _, _, T, FP, _ = frames.shape
            recon_frames = torch.clamp(AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
            assert recon_frames.shape == (S, B, T, FP, FP), "ERROR! unexpected reconstruction shape"
            log_prior = Normal(loc=self.prior_mu, scale=self.prior_std).log_prob(z_what).sum(-1) # S * B * K
            assert log_prior.shape == (S, B, K), "ERROR! unexpected prior shape"
            ll = MBern_log_prob(recon_frames, frames) # S * B * T, log likelihood log p(x | z)

            return log_prior, ll, recon_frames

def MBern_log_prob(x_mean, x, EPS=1e-9):
    """
    the size is ... * H * W
    so I added two sum ops
    """
    return (torch.log(x_mean + EPS) * x +
                torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)
