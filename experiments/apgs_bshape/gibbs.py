import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from combinators import Trace, Program, copytraces, Compose, Propose, Resample, Extend
from experiments.apgs_bshape.models import Enc_coor, Enc_digit, Decoder, apg_ix

loss_tuple = namedtuple('loss_tuple', ['phi', 'theta'])

def loss_is(out, total_loss):
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    log_q = out.proposal_trace.log_joint(**jkwargs)
    log_p = out.target_trace.log_joint(**jkwargs)
    # !!! need this metric as <density>
    loss_phi = (w * (- log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()
    return loss_phi + loss_theta + total_loss

def loss_apg(out, total_loss):    
    foo = [(k,v.provenance) for k,v in out.proposal_trace.items()]
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    # This is a hack to find the marginal of the forward kernel.
    assert out.q1_trace is not None
    if out.q_out.q1_out.type == 'Resample':
        forward_trace = out.q2_trace
    else:
        forward_trace = out.q1_trace
#     from IPython.core.debugger import Tracer; Tracer()()
    log_q = forward_trace.log_joint(**jkwargs)
    log_p = out.target_trace.log_joint(nodes={'recon'}, **jkwargs)
    # !!! need this metric as <density>
    loss_phi = (w * (- log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()
    
    return loss_phi + loss_theta + total_loss
#     return loss_tuple(phi=total_loss.phi + loss_phi, theta=total_loss.theta + loss_theta)

class Noop(Program):
    def __init__(self):
        super().__init__()
    def model(self, trace, c):
        return c


# Implement in a way that it extracts cov_kernel when t=0, and extracts z_where_{t-1} from c
def gibbs_sweeps(models, num_sweeps, T):
    q_enc_coor = models['enc_coor']
    q_enc_digit = models['enc_digit']
    p_dec_os = models['dec']

    # We need this because Enc_coor to swallow first index
    q_os = Noop()
    for t in range(0, T):
        q_os = Compose(q_os, q_enc_coor, ix=apg_ix(t, 0, "forward"))
    q_os = Compose(q_os, q_enc_digit, ix=apg_ix(T, 0, "forward"))

    q_is = Propose(p=p_dec_os, q=q_os, ix=apg_ix(T, 0, "forward"),
                   loss_fn=loss_is)
    q_is = Resample(q_is)

    q_t = q_is
    for sweep in range(1, num_sweeps+1): # Sweeps
        for t in range(T): # Time step
            q_t = Propose(p=Extend(p_dec_os, q_enc_coor, ix=apg_ix(t, sweep, "reverse")),
                          q=Compose(q_t, q_enc_coor, ix=apg_ix(t, sweep, "forward")),
                          loss_fn=loss_apg, ix=apg_ix(t, sweep, "propose"))
            q_t = Resample(q_t)
        q_t = Propose(p=Extend(p_dec_os, q_enc_digit, ix=apg_ix(T, sweep, "reverse")),
                      q=Compose(q_t, q_enc_digit, ix=apg_ix(T, sweep, "forward")),
                      loss_fn=loss_apg, ix=apg_ix(T, sweep, "propose"))
        q_t = Resample(q_t)
    return q_t


