import torch.nn.functional as F


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

    recon_key = (
        "recon"
        if "recon" in out.target_trace
        else "recon_%d_%d" % (out.ix.t, out.ix.sweep)
    )

    log_p = out.target_trace[recon_key].log_prob.sum(-1).sum(-1)
    if len(log_p.shape) == 3:
        log_p = log_p.sum(-1)

    log_q = forward_trace.log_joint(**jkwargs)
    loss_phi = (w * (-log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()

    return loss_phi + loss_theta + total_loss
