import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import math
import torch.nn.functional as F
import combinators.trace.utils as trace_utils

def copy_trace(trace, exclude_name):
    return trace_utils.copysubtrace(trace, set(trace.keys()) - set(exclude_name))

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

class Dec_coor():
    """
    generative model of digit positions
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0) : S * B * D
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, z_where_dim, CUDA, device):
        super(self.__class__, self)

        self.prior_mu0 = torch.zeros(z_where_dim)
        self.prior_Sigma0 = torch.ones(z_where_dim) * 1.0
        self.prior_Sigmat = torch.ones(z_where_dim) * 0.2
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu0  = self.prior_mu0.cuda()
                self.prior_Sigma0 = self.prior_Sigma0.cuda()
                self.prior_Sigmat = self.prior_Sigmat.cuda()
                
    def forward(self, q, p, timestep):
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
        return p 
    
    
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
        
    def forward(self, q, frames, extend_dir):
        z_where = []
        for t in range(frames.shape[2]):
            z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
        z_where = torch.cat(z_where, 2)
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

    
 
        
class Dec_digit(nn.Module):
    """
    decoder of the digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT, CUDA, device):
        super(self.__class__, self).__init__()
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()
        self.AT = AT
        
    def forward(self, q, p, frames, recon_level, timestep=None):
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
        

# def MBern_log_prob(x_mean, x, EPS=1e-9):
#     """
#     the size is ... * H * W
#     so I added two sum ops
#     """
#     return (torch.log(x_mean + EPS) * x +
#                 torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)

