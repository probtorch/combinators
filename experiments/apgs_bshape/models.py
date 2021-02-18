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

def init_models(frame_pixels, shape_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, num_objects, mean_shape, device, reparameterized=False):
    models = dict()
    AT = Affine_Transformer(frame_pixels, shape_pixels, device)

    models['dec'] = Decoder(num_pixels=shape_pixels**2,
                            num_hidden=num_hidden_digit,
                            z_where_dim=z_where_dim,
                            z_what_dim=z_what_dim,
                            AT=AT,
                            device=device,
                            reparameterized=reparameterized).to(device)
    models['enc_coor'] = Enc_coor(num_pixels=(frame_pixels-shape_pixels+1)**2,
                                  mean_shape=mean_shape,
                                  num_hidden=num_hidden_coor,
                                  z_where_dim=z_where_dim,
                                  AT=AT,
                                  dec=models["dec"],
                                  num_objects=num_objects,
                                  reparameterized=reparameterized).to(device)
    models['enc_digit'] = Enc_digit(num_pixels=shape_pixels**2,
                                    num_hidden=num_hidden_digit,
                                    z_what_dim=z_what_dim,
                                    AT=AT,
                                    reparameterized=reparameterized).to(device)
    return models

class Enc_coor(Program):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, AT, dec, mean_shape, num_objects, reparameterized=False):
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
        self.get_dec = lambda : dec
        self.mean_shape = mean_shape
        self.K = num_objects
        self.reparameterized = reparameterized

    def model(self, trace, c, ix):
        frames = c["frames"]
        S, B, T, FP, _ = frames.shape
        if ix.sweep == 0:
            # FIXME: FIgure out if we can use cheaper expand here
            conv_kernel = self.mean_shape.repeat(S, B, self.K, 1, 1)
        else:
            if ix.dir == "forward":
                z_what_val = c['z_what_%d' % (ix.sweep-1)]
            elif ix.dir == "reverse":
                z_what_val = trace._cond_trace['z_what_%d' % (ix.sweep-1)].value
            else:
                raise ValueError("Kernel can only be run forward or in reverse")
            conv_kernel = self.get_dec().get_conv_kernel(z_what_val)

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
                z_where_val_k = Normal(q_mean_k, q_std_k).sample()
                z_where_t.append(z_where_val_k.unsqueeze(2))
            elif ix.dir == 'reverse':
                z_where_val_k = trace._cond_trace['z_where_%d_%d' % (ix.t, ix.sweep-1)].value[:,:,k,:]
            recon_k = self.AT.digit_to_frame(conv_kernel[:,:,k,:,:].unsqueeze(2), z_where_val_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            assert recon_k.shape ==(S,B,FP,FP), 'shape = %s' % recon_k.shape
            frame_left = frame_left - recon_k
        q_mean = torch.cat(q_mean, 2)
        q_std = torch.cat(q_std, 2)

        # Stuff happens in a Compose
        if ix.dir == 'forward':
            z_where_t = torch.cat(z_where_t, 2)
            # For performace reasons we want to add all K objects as one RV, hence we need to cheat here:
            # We sampled all K RVs manually in for-loop above, and "simulate" a combinators sampling operation here.
            # if ix.t <= 1 and ix.sweep > 0:
            #     breakpoint()
            trace.append(RandomVariable(Normal(loc=q_mean, scale=q_std),
                                        value=z_where_t,
                                        provenance=Provenance.SAMPLED,
                                        reparameterized=self.reparameterized),
                         name='z_where_%d_%d'%(ix.t, ix.sweep))
            # We need this because in initial IS step this is not run as a "kernel"
            if ix.sweep == 0:
                return {**c, 'z_where_%d_%d'%(ix.t, ix.sweep): z_where_t}
            return c
        elif ix.dir == 'reverse':
            trace.normal(loc=q_mean,
                         scale=q_std,
                         name='z_where_%d_%d'%(ix.t, ix.sweep-1),
                         reparameterized=self.reparameterized)
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
        self.AT = AT
        self.reparameterized = reparameterized
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
            trace.normal(loc=q_mu, scale=q_std, name='z_what_%d'%(ix.sweep), reparameterized=self.reparameterized)
        elif ix.dir == 'reverse':
            trace.normal(loc=q_mu, scale=q_std, name='z_what_%d'%(ix.sweep-1), reparameterized=self.reparameterized)
        else:
            raise ValueError("Kernel must be run either forward or reverse")

class Decoder(Program):
    """
    decoder
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, z_what_dim, AT, device, reparameterized=False):
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
        self.reparameterized = reparameterized
    # In q_\phi case
    def get_conv_kernel(self, z_what_value, detach=True):
        digit_mean = self.dec_digit_mean(z_what_value)  # S * B * K * (28*28)
        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        digit_mean = digit_mean.detach() if detach else digit_mean
        return digit_mean

    def model(self, trace, c, ix, EPS=1e-9):
        frames = c["frames"]
        _, _, T, FP, _ = frames.shape
        ##################################################################
        # Remove the optimization by always computing the log joint.
        # Compute the log prior of z_where_{1:T} and construct a sample
        # trajectory using sample values
        # 1. from sweep for preceding and current timesteps i.e. t <= ix.t
        # 2. from sweep-1 for future timesteps, i.e. t > ix.t
        ##################################################################
        z_where_vals = []
        for t in range(T):
            if t <= ix.t:
                if t == 0:
                    trace.normal(loc=self.prior_where0_mu,
                                 scale=self.prior_where0_Sigma,
                                 reparameterized=self.reparameterized,
                                 name='z_where_%d_%d' % (0, ix.sweep))
                else:
                    trace.normal(loc=trace._cond_trace['z_where_%d_%d' % (t-1, ix.sweep)].value,
                                 scale=self.prior_wheret_Sigma,
                                 reparameterized=self.reparameterized,
                                 name='z_where_%d_%d' % (t, ix.sweep))
                z_where_vals.append(trace['z_where_%d_%d' % (t, ix.sweep)].value.unsqueeze(2))

            elif t == (ix.t+1):
                trace.normal(loc=trace._cond_trace['z_where_%d_%d' % (ix.t, ix.sweep)].value,
                             scale=self.prior_wheret_Sigma,
                             reparameterized=self.reparameterized,
                             name='z_where_%d_%d' % (t, ix.sweep-1))
                z_where_vals.append(trace['z_where_%d_%d' % (t, ix.sweep-1)].value.unsqueeze(2))
            else:
                trace.normal(loc=trace._cond_trace['z_where_%d_%d' % (t-1, ix.sweep-1)].value,
                             scale=self.prior_wheret_Sigma,
                             reparameterized=self.reparameterized,
                             name='z_where_%d_%d' % (t, ix.sweep-1))
                z_where_vals.append(trace['z_where_%d_%d' % (t, ix.sweep-1)].value.unsqueeze(2))

        # NOTE: this is only used in the z_what case
        z_where_vals = torch.cat(z_where_vals, 2)

        # index for z_what given ix
        if ix.t == T:
            z_what_index = ix.sweep
        elif ix.t < T and ix.sweep > 0:
            z_what_index = ix.sweep - 1
        else:
            raise ValueError('You should not call the decoder when t=%d and sweep=%d' % (ix.t, ix.sweep))

        trace.normal(loc=self.prior_what_mu,
                    scale=self.prior_what_std,
                    reparameterized=self.reparameterized,
                    name='z_what_%d'%(z_what_index))
        z_what_val = trace._cond_trace["z_what_%d"%(z_what_index)].value

        if ix.sweep != 0:
            if ix.t == 0:
                old_recon_name = "recon_%d_%d"%(T, ix.sweep-1)
            else:
                old_recon_name = "recon_%d_%d"%(ix.t-1, ix.sweep)
            old_recon = trace._cond_trace[old_recon_name]
            dummy_zeros = torch.zeros_like(old_recon.log_prob)
            trace.append(
                RandomVariable(
                    dist=Bernoulli(probs=dummy_zeros),
                    value=old_recon.value,
                    log_prob=dummy_zeros,
                    reparameterized=self.reparameterized,
                    can_resample=False,
                    provenance=Provenance.REUSED), # Needs to be reused to be picked up into tau_2, i.e. the weight computation
                name=old_recon_name)

        # optimization for z_wheres
        if ix.t == 0:
            z_where_vals = trace._cond_trace["z_where_%d_%d" % (0, ix.sweep)].value.unsqueeze(2)
            digit_mean = self.get_conv_kernel(z_what_val, detach=False)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where_vals).sum(-3), min=0.0, max=1.0).squeeze(2)

            trace.variable(
                Bernoulli,
                probs=recon_frames,
                value=frames[:,:,ix.t,:,:],
                name='recon_%d_%d' % (0, ix.sweep),
                reparameterized=self.reparameterized,
                provenance=Provenance.OBSERVED)

            recon_optimizing_denominator = trace._cond_trace['recon_%d_%d' % (T, ix.sweep-1)].log_prob[:,:,1:,:,:]
            opt_fake_likelihood = (-1) * recon_optimizing_denominator

            trace.append(
                RandomVariable(
                    dist=Bernoulli(probs=recon_frames),
                    value=frames[:,:,ix.t,:,:],
                    log_prob=opt_fake_likelihood,
                    reparameterized=self.reparameterized,
                    can_resample=False,
                    provenance=Provenance.REUSED),
                name='recon_opt_%d_%d' % (ix.t, ix.sweep))

        elif ix.t > 0 and ix.t < T:
            # z_where case
            z_where_val_keys = [
                # current step, current sweep
                "z_where_%d_%d" % (ix.t, ix.sweep),
                # current step, old sweep
                "z_where_%d_%d" % (ix.t, ix.sweep-1) # <<< want to add this to the denominator
            ]
            z_where_vals = [trace._cond_trace[k].value.unsqueeze(2) for k in z_where_val_keys]
            z_where_vals = torch.cat(z_where_vals, 2)

            digit_mean = self.get_conv_kernel(z_what_val, detach=False)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where_vals).sum(-3), min=0.0, max=1.0)

            # current z_where reconstruction likelihood.
            _ = trace.variable(
                Bernoulli,
                probs=recon_frames[:,:,0],
                value=frames[:,:,ix.t,:,:],
                reparameterized=self.reparameterized,
                name='recon_%d_%d' % (ix.t, ix.sweep),
                provenance=Provenance.OBSERVED)

            # manual reconstruction ratio for z_where_t=ix.t_s=ix.sweep-1
            recon_optimizing_denominator = Bernoulli(probs=recon_frames[:,:,1]).log_prob(frames[:,:,1])

            recon_numerator_key = 'recon_%d_%d' % (ix.t-1, ix.sweep)
            recon_optimizing_numerator = trace._cond_trace[recon_numerator_key].log_prob

            opt_fake_likelihood = recon_optimizing_numerator - recon_optimizing_denominator

            trace.append(
                RandomVariable(
                    dist=Bernoulli(probs=recon_frames),
                    value=frames[:,:,ix.t,:,:],
                    log_prob=opt_fake_likelihood,
                    reparameterized=self.reparameterized,
                    can_resample=False,
                    provenance=Provenance.REUSED),
                name='recon_opt_%d_%d' % (ix.t, ix.sweep))

        elif ix.t == T:
            # z_what case + IS
            if ix.sweep != 0:
                z_where_vals_denominator = [trace['z_where_%d_%d' % (t, ix.sweep)].value.unsqueeze(2) for t in range(T-1)]
                z_where_vals_denominator = torch.cat(z_where_vals_denominator, 2)
                z_what_val_denominator = trace._cond_trace['z_what_%d' % (ix.sweep-1)].value

                digit_mean = self.get_conv_kernel(z_what_val_denominator, detach=False)
                recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where_vals_denominator).sum(-3), min=0.0, max=1.0)
                recon_optimizing_denominator = Bernoulli(probs=recon_frames).log_prob(frames[:,:,:T-1])
                opt_fake_likelihood = (-1) * recon_optimizing_denominator

                trace.append(
                    RandomVariable(
                        dist=Bernoulli(probs=recon_frames),
                        value=frames[:,:,:T-1,:,:],
                        log_prob=opt_fake_likelihood,
                        reparameterized=self.reparameterized,
                        can_resample=False,
                        provenance=Provenance.REUSED),
                    name='recon_opt_%d_%d' % (ix.t, ix.sweep))

            # Numerator section
            digit_mean = self.get_conv_kernel(z_what_val, detach=False)
            recon_frames = torch.clamp(self.AT.digit_to_frame(digit_mean, z_where_vals).sum(-3), min=0.0, max=1.0)

            _ = trace.variable(Bernoulli,
                               probs=recon_frames,
                               value=frames,
                               name='recon_%d_%d' % (T, ix.sweep),
                               reparameterized=self.reparameterized,
                               provenance=Provenance.OBSERVED)


        return {**{"z_what_%d"%(z_what_index): z_what_val, "frames": c["frames"]},
                **{"z_where_%d_%d"%(t, ix.sweep): trace._cond_trace['z_where_%d_%d' % (t, ix.sweep)].value for t in range(min(ix.t+1, T))}}
