import torch.nn.functional as F
from combinators import Program, Compose, Propose, Resample, Extend
from experiments.apgs_bshape.models import apg_ix


def loss_is(out, total_loss):
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)
    log_q = out.proposal_trace.log_joint(**jkwargs)
    log_p = out.target_trace.log_joint(**jkwargs)
    loss_phi = (w * (-log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()
    return loss_phi + loss_theta + total_loss


def loss_apg(out, total_loss):
    jkwargs = dict(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = out.log_weight.detach()
    w = F.softmax(log_w, 0)

    # This is a hack to find the marginal of the forward kernel.
    assert out.forward_trace is not None
    forward_trace = out.forward_trace

    recon_key = "recon" if "recon" in out.target_trace else \
        "recon_%d_%d" % (out.ix.t, out.ix.sweep)

    log_p = out.target_trace[recon_key].log_prob.sum(-1).sum(-1)
    if len(log_p.shape) == 3:
        log_p = log_p.sum(-1)

    log_q = forward_trace.log_joint(**jkwargs)
    loss_phi = (w * (-log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()

    return loss_phi + loss_theta + total_loss


class Noop(Program):
    def __init__(self):
        super().__init__()

    def model(self, trace, c):
        return c


# Implement in a way that it extracts cov_kernel when t=0, and extracts z_where_{t-1} from c
def gibbs_sweeps(models, num_sweeps, T):
    q_enc_coor = models["enc_coor"]
    q_enc_digit = models["enc_digit"]
    p_dec_os = models["dec"]

    fwd_ix = lambda t, s: apg_ix(t, s, "forward")
    rev_ix = lambda t, s: apg_ix(t, s, "reverse")
    prp_ix = lambda t, s: apg_ix(t, s, "propose")
    _no_reruns = True

    # We need this because Enc_coor to swallow first index
    q_os = Noop()
    for t in range(0, T):
        q_os = Compose(q1=q_os, q2=q_enc_coor, ix=fwd_ix(t, 0))
    q_os = Compose(q1=q_os, q2=q_enc_digit, ix=fwd_ix(T, 0))

    q_is = Propose(
        p=p_dec_os, q=q_os, ix=fwd_ix(T, 0), _no_reruns=_no_reruns, loss_fn=loss_is
    )

    if num_sweeps > 0:
        q_is = Resample(q_is, normalize_weights=True)

    q_t = q_is
    for sweep in range(1, num_sweeps + 1):  # Sweeps
        for t in range(T):  # Time step
            q_t = Propose(
                p=Extend(p_dec_os, q_enc_coor, ix=rev_ix(t, sweep)),
                q=Compose(q1=q_t, q2=q_enc_coor, _debug=True, ix=fwd_ix(t, sweep)),
                _no_reruns=_no_reruns,
                loss_fn=loss_apg,
                ix=prp_ix(t, sweep),
            )
            q_t = Resample(q_t, normalize_weights=True)
        q_t = Propose(
            p=Extend(p_dec_os, q_enc_digit, ix=rev_ix(T, sweep)),
            q=Compose(q1=q_t, q2=q_enc_digit, _debug=True, ix=fwd_ix(T, sweep)),
            _no_reruns=_no_reruns,
            loss_fn=loss_apg,
            ix=prp_ix(T, sweep),
        )
        if sweep != num_sweeps:
            q_t = Resample(q_t, normalize_weights=True)
    return q_t
