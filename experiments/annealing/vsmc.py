from combinators import Propose, Extend, Compose, Resample
from experiments.annealing.objectives import stl_trace


def vsmc(targets, forwards, reverses, loss_fn, resample=False):
    q = targets[0]
    for k, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = Propose(
            p=Extend(p, rev),
            q=Compose(fwd, q, _debug=True),
            loss_fn=lambda out, loss: loss_fn(out) + loss,
            ix=k,
            _no_reruns=False,
            _debug=True,
            transf_q_trace=None if loss_fn.__name__ == "nvo_avo" else stl_trace,
        )
        if resample and k < len(forwards) - 1:
            q = Resample(q)
    return q
