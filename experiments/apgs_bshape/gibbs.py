import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from combinators import Trace, Program, copytraces, Compose, Propose, Resample, Extend
from experiments.apgs_bshape.models import Enc_coor, Enc_digit, Decoder, apg_ix

loss_tuple = namedtuple('loss_tuple', ['phi', 'theta'])

def loss_fn(out, total_loss):
    return 0.
    ix = out.p_out.ix
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)

    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)

    # out <- (output, trace, lw, extra_stuff)
    # out.q_out
    # We provide: out.p_num, out.q_den

    log_q = out.q_log_prob if ix.block == "is" else \
        out.q_log_prob - out.q_out.program.trace.log_joint(**jkwargs)
    log_p = out.p_log_prob
    # !!! need this metric as <density>

    loss_phi = (w * (- log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()

    return loss_tuple(phi=total_loss.phi + loss_phi, theta=total_loss.theta + loss_theta)


# Implement in a way that it extracts cov_kernel when t=0, and extracts z_where_{t-1} from c
def gibbs_sweeps(models, sweeps, T):
    q_enc_coor = models['enc_coor']
    q_enc_digit = models['enc_digit']
    p_dec_os = models['dec']

    q_os = q_enc_coor
    for t in range(T):
        q_os = Compose(q_enc_coor, q_os, ix=apg_ix(t, 0, "forward"))
    q_os = Compose(q_enc_digit, q_os, ix=apg_ix(T, 0, "forward"))

    q_is = Propose(p=p_dec_os, q=q_os, loss_fn=loss_fn)
    q_is = Resample(q_is)

    q_t = q_is
    for sweep in (1, sweeps+1): # Sweeps
        for t in range(T): # Time step
            q_t = Propose(p=Extend(p_dec_os, q_enc_coor, ix=apg_ix(t, sweep, "reverse")),
                          q=Compose(q_enc_coor, q_t, ix=apg_ix(t, sweep, "forward"),
                          loss_fn=loss_fn)
            q_t = Resample(q_t)
        q_t = Propose(p=Extend(p_dec_os, q_enc_digit, ix=apg_ix(t=T, sweep, "reverse")),
                      q=Compose(q_enc_digit, q_t, ix=apg_ix(t=T, sweep, "forward")),
                      loss_fn=loss_fn)
        q_t = Resample(q_t)
    return q_t


