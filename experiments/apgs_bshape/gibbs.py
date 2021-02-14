import math
import torch
import torch.nn as nn
from combinators.stochastic import Trace
from combinators.program import Program
from combinators.trace.utils import copytra
from combinators.experiments.models import Enc_coor, Enc_digit, Dec_os


def loss_fn(out, total_loss):
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

    return losstpl(phi=total_loss.phi + loss_phi, theta=total_loss.theta + loss_theta)


# Implement in a way that it extracts cov_kernel when t=0, and extracts z_where_{t-1} from c
def gibbs_sweeps(K, T)
    q_enc_coor = Enc_coor()
    q_enc_digit = Enc_digit()
    p_dec_os = Dec_os()

    q_os = q_enc_coor
    for t in range(1, T):
        q_os = Compose(q_os, q_enc_coor, idx=t, dir="forward")
    q_os = Compose(q_is, q_enc_digit, dir="forward")

    q_is = Propose(target=p_dec_os,
                   proposal=q_os, loss_fn=loss_fn)
    q_is = Resample(q_is)

    q_t = q_is
    for k in K: # Sweeps
        for t in range(T): # Time step
            q_t = Propose(target=Extend(p_dec_os, q_enc_coor, idx=t, dir="reverse"),
                          proposal=Compose(q_t, q_enc_coor, idx=t, dir="forward"),
                          loss_fn=loss_fn)
            q_t = Resample(q_t)
        q_t = Propose(target=Extend(p_dec_os, q_enc_digit, dir="reverse"),
                      proposal=Compose(q_t, q_enc_digit, dir="forward"),
                      loss_fn=loss_fn)
        q_t = Resample(q_t)

