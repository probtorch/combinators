import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from combinators.stochastic import Trace, Provenance, RandomVariable
from combinators.program import Program
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from experiments.apgs_bshape.affine_transformer import Affine_Transformer
from collections import namedtuple

apg_ix = namedtuple("apg_ix", ["t", "sweep", "dir"])

# from combinators.inference import Program, Compose, Extend, Propose, Resample, Condition
# Path to example programs: 

def init_models(frame_pixels, shape_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, K, mean_shape, device):
    models = dict()
    AT = Affine_Transformer(frame_pixels, shape_pixels, device)

    models['dec'] = Decoder(num_pixels=shape_pixels**2,
                            num_hidden=num_hidden_digit,
                            z_where_dim=z_where_dim,
                            z_what_dim=z_what_dim,
                            AT=AT,
                            device=device).to(device)
    models['enc_coor'] = Enc_coor(num_pixels=(frame_pixels-shape_pixels+1)**2,
                                  mean_shape=mean_shape,
                                  num_hidden=num_hidden_coor,
                                  z_where_dim=z_where_dim,
                                  AT=AT,
                                  dec=models["dec"],
                                  K=K).to(device)
    models['enc_digit'] = Enc_digit(num_pixels=shape_pixels**2,
                                    num_hidden=num_hidden_digit,
                                    z_what_dim=z_what_dim,
                                    AT=AT).to(device)
    return models

class Enc_coor(Program):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, dec, mean_shape, K):
        super(self.__class__, self).__init__()
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
        self.AT = AT
        self.dec = dec
        self.mean_shape = mean_shape
        self.K = K

    def model(self, trace, c, ix):
        breakpoint()
        frames = c["frames"]
        S, B, T, FP, _ = frames.shape
        if ix.sweep == 0:
            # FIXME: FIgure out if we can use cheaper expand here
            conv_kernel = self.mean_shape.repeat(S, B, self.K, 1, 1)
        else:
            z_what = c['z_what_%d' % (ix.sweep-1)]
            conv_kernel = self.dec.get_conv_kernel(z_what)

        _, _, K, DP, _ = conv_kernel.shape
        frame_left = frames[:,:,ix.t,:,:]
        q_mean, q_std = [], []
        z_where_t = []
        # K objects in frame
        for k in range(self.K):
            conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), conv_kernel[:,:,k,:,:].view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            hidden = self.enc_coor_hidden(conved_k)
            q_mean_k = self.where_mean(hidden)
            q_std_k = self.where_log_std(hidden).exp()
            q_mean.append(q_mean_k.unsqueeze(2))

            q_std.append(q_std_k.unsqueeze(2))
            if ix.dir == 'forward':
                z_where_k = Normal(q_mean_k, q_std_k).sample()
                z_where_t.append(z_where_k.unsqueeze(2))
            elif ix.dir == 'reverse':
                z_where_k = q['z_where_%d' % (ix.t+1)].value[:,:,k,:]
            recon_k = self.AT.digit_to_frame(conv_kernel[:,:,k,:,:].unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            assert recon_k.shape ==(S,B,FP,FP), 'shape = %s' % recon_k.shape
            frame_left = frame_left - recon_k
        q_mean = torch.cat(q_mean, 2)
        q_std = torch.cat(q_std, 2)

        # Stuff happens in a Compose
        if ix.dir == 'forward':
            z_where_t = torch.cat(z_where_t, 2)
            # For performace reasons we want to add all K objects as one RV, hence we need to cheat here:
            # We sampled all K RVs manually in for-loop above, and "simulate" a combinators sampling operation here.
            trace.append(RandomVariable(Normal(loc=q_mean, scale=q_std),
                                        value=z_where_t,
                                        provenance=Provenance.SAMPLED,
                                        reparameterized=False),
                         name='z_where_%d_%d'%(ix.t, ix.sweep))
            # We need this because in initial IS step this is not run as a "kernel"
            if ix.sweep == 0:
                return {**c, 'z_where_%d_%d'%(ix.t, ix.sweep): z_where_t}
            return c
        elif ix.dir == 'reverse':
            trace.normal(loc=q_mean, scale=q_std, name='z_where_%d_%d'%(ix.t, ix.sweep-1))
        else:
            raise ValueError("Kernel must be run either forward or reverse")

class Enc_digit(Program):
    """
    encoder of digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT):
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
        self.AT = AT

    def model(self, trace, c, ix):
        frames = c["frames"]
        # z_where are fetched from the input in the for loop below

        z_where = []
        for t in range(frames.shape[2]):
            z_where.append(c["z_where_%d_%d"%(t, ix.sweep)].unsqueeze(2))
        z_where = torch.cat(z_where, 2)
        cropped = self.AT.frame_to_digit(frames=frames, z_where=z_where)
        cropped = torch.flatten(cropped, -2, -1)
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden)
        q_std = self.enc_digit_log_std(hidden).exp()
        if ix.dir == 'forward':
            trace.normal(loc=q_mu, scale=q_std, name='z_what_%d'%(ix.sweep))
        elif ix.dir == 'reverse':
            trace.normal(loc=q_mu, scale=q_std, name='z_what_%d'%(ix.sweep-1))
        else:
            raise ValueError

class Decoder(Program):
    """
    decoder
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, z_what_dim, AT, device):
        super(self.__class__, self).__init__()
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_where0_mu = torch.zeros(z_where_dim, device=device)
        self.prior_where0_Sigma = torch.ones(z_where_dim, device=device) * 1.0
        self.prior_wheret_Sigma = torch.ones(z_where_dim, device=device) * 0.2
        self.prior_what_mu = torch.zeros(z_what_dim, device=device)
        self.prior_what_std = torch.ones(z_what_dim, device=device)
        self.AT = AT

    # In q_\phi case
    def get_conv_kernel(self, z_what, detach=True):
        digit_mean = self.dec_digit_mean(z_what)  # S * B * K * (28*28)
        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        digit_mean = digit_mean.detach() if detach else digit_mean
        return digit_mean

    def model(self, trace, c, ix):
        frames = c["frames"]
        _, _, T, FP, _ = frames.shape

        # In p_\theta - z_where case
        if ix.sweep > 0 and ix.t != T:
            digit_mean = self.get_conv_kernel(trace._cond_trace["z_what_%d"%(ix.sweep-1)], detach=False)
            # prior of z_where
            if ix.t == 0:
                trace.normal(loc=self.prior_where0_mu,
                             scale=self.prior_where0_Sigma,
                             name='z_where_%d_%d' % (ix.t, ix.sweep))

            else:
                trace.normal(loc=trace._cond_trace['z_where_%d_%d' % (ix.t-1, ix.sweep)].value,
                             scale=self.prior_wheret_Sigma,
                             name='z_where_%d_%d' % (ix.t, ix.sweep))

            if ix.t < (T-1):
                trace.normal(loc=trace._cond_trace['z_where_%d' % (ix.t, ix.sweep)].value,
                             scale=self.prior_wheret_Sigma,
                             name='z_where_%d_%d' % (ix.t+1, ix.sweep-1))

            z_where = trace._cond_trace['z_where_%d' % (ix.t, ix.sweep)].value
            # prior of z_what
            trace.normal(loc=self.prior_what_mu,
                         scale=self.prior_what_std,
                         name='z_what_%d'%(ix.sweep-1))
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where.unsqueeze(2)).squeeze(2).sum(-3), min=0.0, max=1.0) # S * B * FP * FP
            _ = trace.variable(Bernoulli, probs=recon_frames, value=frames[:,:,ix.t,:,:], name='recon',
                               provenance=Provenance.OBSERVED)
            return {"z_where_%d_%d"%(ix.t, ix.sweep): z_where, "frames": c["frames"]}

        # In p_\theta - z_what case
        # For z_what we need to reconstruct all frames
        else:
            z_what = trace._cond_trace["z_what_%d"%(ix.sweep)]
            digit_mean = self.get_conv_kernel(z_what, detach=False)
            # prior of z_where
            z_wheres = []
            for t in range(T):
                if t == 0:
                    trace.normal(loc=self.prior_where0_mu,
                                 scale=self.prior_where0_Sigma,
                                 name='z_where_%d_%d' % (t, ix.sweep))
                else:
                    trace.normal(loc=q['z_where_%d_%d' % (t-1, ix.sweep)].value,
                                 scale=self.prior_wheret_Sigma,
                                 name='z_where_%d_%d' % (t, ix.sweep))
                z_wheres.append(trace['z_where_%d_%d' % (t, ix.sweep)].value.unsqueeze(2))
            # prior of z_what
            trace.normal(loc=self.prior_what_mu,
                         scale=self.prior_what_std,
                         name='z_what_%d'%(ix.sweep))
            z_wheres = torch.cat(z_wheres, 2)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_wheres).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
            _= trace.variable(Bernoulli, probs=recon_frames, value=frames, name='recon', provenance=Provenance.OBSERVED)
            return {"z_what_%d"%(ix.sweep): z_what, "frames": c["frames"]}
