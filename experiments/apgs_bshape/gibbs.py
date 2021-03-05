import torch.nn.functional as F
from collections import namedtuple
from combinators import Program, Compose, Propose, Resample, Extend
from experiments.apgs_bshape.models import apg_ix

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
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    # This is a hack to find the marginal of the forward kernel.
    assert out.forward_trace is not None
    forward_trace = out.forward_trace

    log_p = out.target_trace['recon_%d_%d'%(out.ix.t, out.ix.sweep)].log_prob.sum(-1).sum(-1)
    if len(log_p.shape) == 3:
        log_p = log_p.sum(-1)

    log_q = forward_trace.log_joint(**jkwargs)
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
                   _no_reruns=True,
                   loss_fn=loss_is)

    if num_sweeps > 0:
        q_is = Resample(q_is, quiet=True, normalize_weights=True)

    q_t = q_is
    for sweep in range(1, num_sweeps+1): # Sweeps
        for t in range(T): # Time step
            q_t = Propose(p=Extend(p_dec_os, q_enc_coor, ix=apg_ix(t, sweep, "reverse")),
                          q=Compose(q_t, q_enc_coor, ix=apg_ix(t, sweep, "forward")),
                          _no_reruns=True,
                          loss_fn=loss_apg, ix=apg_ix(t, sweep, "propose"))
            q_t = Resample(q_t, quiet=True, normalize_weights=True)
        q_t = Propose(p=Extend(p_dec_os, q_enc_digit, ix=apg_ix(T, sweep, "reverse")),
                      q=Compose(q_t, q_enc_digit, ix=apg_ix(T, sweep, "forward")),
                      _no_reruns=True,
                      loss_fn=loss_apg, ix=apg_ix(T, sweep, "propose"))
        if sweep != num_sweeps:
            q_t = Resample(q_t, quiet=True, normalize_weights=True)
    return q_t


