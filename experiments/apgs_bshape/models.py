import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from combinators.stochastic import Trace, Provenance
from combinators.program import Program
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from experiments.apgs_bshape.affine_transformer import Affine_Transformer

# from combinators.inference import Program, Compose, Extend, Propose, Resample, Condition
# Path to example programs: 

def init_models(frame_pixels, shape_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, device):
    models = dict()
    AT = Affine_Transformer(frame_pixels, shape_pixels, device)

    models['dec'] = Decoder(num_pixels=shape_pixels**2,
                            num_hidden=num_hidden_digit,
                            z_where_dim=z_where_dim,
                            z_what_dim=z_what_dim,
                            AT=AT,
                            device=device).cuda().to(device)
    models['enc-coor'] = Enc_coor(num_pixels=(frame_pixels-shape_pixels+1)**2,
                                  num_hidden=num_hidden_coor,
                                  z_where_dim=z_where_dim,
                                  AT=AT,
                                  dec=models["dec"]).cuda().to(device)
    models['enc-digit'] = Enc_digit(num_pixels=shape_pixels**2,
                                    num_hidden=num_hidden_digit,
                                    z_what_dim=z_what_dim,
                                    AT=AT).cuda().to(device)
    return models

class Enc_coor(Program):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, dec, reparameterized=False):
        super(self.__class__, self).__init__()
        self.dec = dec
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
#         self.conv_kernel = mean_shape
        self.AT = AT

    def model(self, trace, c):
        # unpack inputs
        frames = c["frames"]
        timestep = c["timestep"]
        conv_kernel = c["conv_kernel"]
        kernel_dir = c["kernel_dir"]

        # rethink q
        _, _, K, DP, _ = conv_kernel.shape
        S, B, T, FP, _ = frames.shape
        frame_left = frames[:,:,timestep,:,:]
        q_mean, q_std = [], []
        z_where_t = []
        # K objects in frame
        for k in range(K):
            conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), conv_kernel[:,:,k,:,:].view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            hidden = self.enc_coor_hidden(conved_k)
            q_mean_k = self.where_mean(hidden)
            q_std_k = self.where_log_std(hidden).exp()
            q_mean.append(q_mean_k.unsqueeze(2))

            q_std.append(q_std_k.unsqueeze(2))
            if kernel_dir == 'forward':
                z_where_k = Normal(q_mean_k, q_std_k).sample()
                z_where_t.append(z_where_k.unsqueeze(2))
            elif kernel_dir == 'reverse':
                z_where_k = q['z_where_%d' % (timestep+1)].value[:,:,k,:]
            recon_k = self.AT.digit_to_frame(conv_kernel[:,:,k,:,:].unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            assert recon_k.shape ==(S,B,FP,FP), 'shape = %s' % recon_k.shape
            frame_left = frame_left - recon_k
        q_mean = torch.cat(q_mean, 2)
        q_std = torch.cat(q_std, 2)

        # Stuff happens in a Compose
        if kernel_dir == 'forward':
            z_where_t = torch.cat(z_where_t, 2)
            # For performace reasons we want to add all K objects as one RV, hence we need to cheat here:
            # We sampled all K RVs manually in for-loop above, and "simulate" a combinators sampling operation here.
            name='z_where_%d_prime'%(timestep+1)
            trace.normal(loc=q_mean, scale=q_std, value=z_where_t, name=name, provenance=Provenance.SAMPLED)
            name_mod="_".join(name.split("_")[:3])
            return {name_mod: trace[name].value}
        # Stuff happens in a Cond. Extend
        elif kernel_dir == 'reverse':
            trace.normal(loc=q_mean, scale=q_std, name='z_where_%d'%(timestep+1))
            # Don't output anything
        else:
            raise ValueError("Kernel must be run either forward or reverse")

class Enc_digit(Program):
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

    def model(self, trace, kernel_dir):
        frames = c["frame"]
        z_where = []
        for t in range(frames.shape[2]):
            z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
        z_where = torch.cat(z_where, 2)
        cropped = self.AT.frame_to_digit(frames=frames, z_where=z_where)
        cropped = torch.flatten(cropped, -2, -1)
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden)
        q_std = self.enc_digit_log_std(hidden).exp()
        if kernel_dir == 'forward':
            trace.normal(loc=q_mu, scale=q_std, name='z_what_prime')
            return {"z_what": trace["z_what_prime"]}
        elif kernel_dir == 'reverse':
            trace.normal(loc=q_mu, scale=q_std, name='z_what')
        else:
            raise ValueError

class Decoder(Program):
    """
    decoder 
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, z_what_dim, AT, conv_kernel, device):
        super(self.__class__, self).__init__()
        self.conv_kernel = conv_kernel
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_where0_mu = torch.zeros(z_where_dim)
        self.prior_where0_Sigma = torch.ones(z_where_dim) * 1.0
        self.prior_wheret_Sigma = torch.ones(z_where_dim) * 0.2
        self.prior_what_mu = torch.zeros(z_what_dim)
        self.prior_what_std = torch.ones(z_what_dim)

        if torch.cuda.is_available():
            with torch.cuda.device(device):
                self.prior_where0_mu  = self.prior_where0_mu.cuda()
                self.prior_where0_Sigma = self.prior_where0_Sigma.cuda()
                self.prior_wheret_Sigma = self.prior_wheret_Sigma.cuda()
                self.prior_what_mu = self.prior_what_mu.cuda()
                self.prior_what_std = self.prior_what_std.cuda()
        self.AT = AT

    def model(self, trace, c, timestep=None):
        frames = c["frame"]
        recon_level = c["recon_level"]

        if not (recon_level=="object" and "z_what" not in c):
            p = probtorch.Trace()
            digit_mean = self.dec_digit_mean(q['z_what'].value)  # S * B * K * (28*28)
            S, B, K, DP2 = digit_mean.shape
            DP = int(math.sqrt(DP2))
            digit_mean = digit_mean.view(S, B, K, DP, DP)

        if recon_level == 'object': ## return the recnostruction of objects
            if "z_what" in c:
                return {"digit_mean": digit_mean.detach()}
            else:
                return {"conv_kernel": self.conv_kernel}

        elif recon_level =='frame': # return the reconstruction of a single frame
            _, _, T, FP, _ = frames.shape
            # prior of z_where
            if timestep == 0:
                trace.normal(loc=self.prior_where0_mu,
                             scale=self.prior_where0_Sigma,
                             # value=q['z_where_%d' % (timestep+1)].value,
                             name='z_where_%d' % (timestep+1))

            else:
                trace.normal(loc=q['z_where_%d' % (timestep)].value,
                             scale=self.prior_wheret_Sigma,
                             # value=q['z_where_%d' % (timestep+1)].value,
                             name='z_where_%d' % (timestep+1))

            if timestep < (T-1):
                trace.normal(loc=q['z_where_%d' % (timestep+1)].value,
                             scale=self.prior_wheret_Sigma,
                             # value=q['z_where_%d' % (timestep+2)].value,
                             name='z_where_%d' % (timestep+2))

            z_where = q['z_where_%d' % (timestep+1)].value.unsqueeze(2)
            # prior of z_what
            trace.normal(loc=self.prior_what_mu,
                         scale=self.prior_what_std,
                         # value=q['z_what'].value,
                         name='z_what')
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where).squeeze(2).sum(-3), min=0.0, max=1.0) # S * B * FP * FP
            _ = trace.variable(Bernoulli, probs=recon_frames, value=frames[:,:,timestep,:,:], name='recon',
                               provenance=Provenance.OBSERVED)

        elif recon_level =='frames': # return the reconstruction of the entire frames
            _, _, T, FP, _ = frames.shape
            z_wheres = []
            # prior of z_where
            for t in range(T):
                if t == 0:
                    trace.normal(loc=self.prior_where0_mu,
                                 scale=self.prior_where0_Sigma,
                                 # value=q['z_where_%d' % (t+1)].value,
                                 name='z_where_%d' % (t+1))
                else:
                    trace.normal(loc=q['z_where_%d' % (t)].value,
                                 scale=self.prior_wheret_Sigma,
                                 value=q['z_where_%d' % (t+1)].value,
                                 name='z_where_%d' % (t+1))
                z_wheres.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
            # prior of z_what
            trace.normal(loc=self.prior_what_mu,
                         scale=self.prior_what_std,
                         value=q['z_what'].value,
                         name='z_what')
            z_wheres = torch.cat(z_wheres, 2)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_wheres).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
            _= trace.variable(Bernoulli, probs=recon_frames, value=frames, name='recon', provenance=Provenance.OBSERVED)
        else:
            raise ValueError

    # def log_prior(self, frames, z_where, z_what):
    #     T = z_where.shape[2]
    #     for t in range(T):
    #         if t == 0:
    #             log_p = Normal(loc=self.prior_where0_mu, scale=self.prior_where0_Sigma).log_prob(z_where[:,:,t,:,:]).sum(-1).sum(-1)
    #         else:
    #             log_p += Normal(loc=z_where[:,:,t-1,:,:], scale=self.prior_wheret_Sigma).log_prob(z_where[:,:,t,:,:]).sum(-1).sum(-1)

    #     digit_mean = self.dec_digit_mean(z_what)  # S * B * K * (28*28)
    #     S, B, K, DP2 = digit_mean.shape
    #     DP = int(math.sqrt(DP2))
    #     digit_mean = digit_mean.view(S, B, K, DP, DP)
    #     recon_frames = torch.clamp(self.AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
    #     log_p = Normal(self.prior_what_mu,self.prior_what_std).log_prob(z_what).sum(-1).sum(-1)
    #     log_p += Bernoulli(probs=recon_frames).log_prob(frames).sum(-1).sum(-1).sum(-1)
    #     return log_p
