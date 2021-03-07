#!/usr/bin/env python3

''' TODO: encode these into more general tests '''
from experiments.annealing.models import *
from experiments.annealing.main import *
from experiments.annealing.objectives import *

def test_K_step_nvi(K, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False, num_seeds=10):
    out = paper_model(K)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    seeds = torch.arange(num_seeds)
    for seed in seeds:
        torch.manual_seed(seed)
        out = nvi_declarative(targets, forwards, reverses, nvo_avo,
                              sample_shape, batch_dim=batch_dim, sample_dims=sample_dims,
                              resample=resample)
        stats_nvi_run = get_stats(out)

        torch.manual_seed(seed)
        lw = 0.
        cond_trace = None
        cond_traces = [cond_trace]
        out_refs = [out]
        target_outs = []
        proposal_outs = []
        fwd_outs = []
        rev_outs = []
        lws_ = []
        lws = []
        for k in range(K):
            targets[k]._cond_trace = copytraces(cond_trace) if cond_trace is not None else None
            nvi_out  = compute_nvi_weight(targets[k], targets[k+1], forwards[k], reverses[k], lw_in=lw,
                                          sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim,
                                          resample=resample)
            lw, lw_, cond_trace, target_out, proposal_out, fwd_out, rev_out = nvi_out
            proposal_trace_manual = copytraces(fwd_out.trace, proposal_out.trace)
            target_trace_manual = copytraces(rev_out.trace, target_out.trace)

            lws.append(lw)
            lws_.append(lw_)
            cond_traces.append(cond_trace)
            target_outs.append(target_out)
            proposal_outs.append(proposal_out)
            fwd_outs.append(fwd_out)
            rev_outs.append(rev_out)
            if resample:
                out_refs.append(out_refs[-1].q_out.q_out.q1_out)
            else:
                out_refs.append(out_refs[-1].q_out.q1_out)

            # Check weights, proposal_trace, and, target_trace at every step
            assert (valeq(proposal_trace_manual, stats_nvi_run["proposal_trace"][-(k+1)])), "proposal_traces not equal {}".format(k)
            assert (valeq(target_trace_manual, stats_nvi_run["target_trace"][-(k+1)])), "proposal_traces not equal {}".format(k)
            assert (lw_ == stats_nvi_run["lw"][-(k+1)]).all(), "log weights not equal {}".format(k)

def test_nvi_sampling_scheme(num_seeds=100):
    print(f"Testing NVI sampling scheme without resampling (num_seeds: {num_seeds})")
    test_K_step_nvi(8, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=True, num_seeds=num_seeds)
    print(f"Testing NVI sampling scheme with resampling (num_seeds: {num_seeds})")
    test_K_step_nvi(8, sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False, num_seeds=num_seeds)
def compute_nvi_weight(proposal, target, forward, reverse, lw_in=0., sample_shape=(10, 5), batch_dim=1, sample_dims=0, resample=False):
    g0_out = proposal(None, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    g0_rv = g0_out.trace[proposal.name]
    g0_lp = g0_rv.log_prob

    f12_out = forward({f'{proposal.name}': g0_rv.value}, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    f12_rv = f12_out.trace[target.name]
    f12_lp = f12_rv.log_prob

    target._cond_trace = f12_out.trace
    g1_out = target(None, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    g1_rv = g1_out.trace[target.name]
    g1_lp = g1_rv.log_prob

    reverse._cond_trace = g0_out.trace
    r21_out = reverse({f'{target.name}': g1_rv.value}, sample_dims=sample_dims, sample_shape=sample_shape, batch_dim=batch_dim)
    r21_rv = r21_out.trace[proposal.name]
    r21_lp = r21_rv.log_prob

    lw_out = (g1_lp + r21_lp) - (g0_lp + f12_lp) + lw_in
    lw_out_ = lw_out
    trace_out = g1_out.trace
    if resample:
        trace_out, lw_out = Systematic()(trace_out, lw_out, sample_dims=sample_dims, batch_dim=batch_dim)
    return lw_out, lw_out_, trace_out, g1_out, g0_out, f12_out, r21_out

def test_1_step_nvi(sample_shape=(10, 5), batch_dim=1, sample_dims=0):
    out = paper_model(2)
    targets, forwards, reverses = [[m.to(autodevice()) for m in out[n]] for n in ['targets', 'forwards', 'reverses']]

    q = nvi_declarative(targets, forwards, reverses, nvo_avo, sample_shape, batch_dim=batch_dim, sample_dims=sample_dims)
    seeds = torch.arange(100)
    for seed in seeds:
        torch.manual_seed(seed)
        out = q(None, sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
        torch.manual_seed(seed)

        lw, _ = compute_nvi_weight(targets[0], targets[1], forwards[0], reverses[0], lw_in=0.,
                                   sample_shape=sample_shape, sample_dims=sample_dims, batch_dim=batch_dim)
        assert (lw == out.log_weight).all()

    # main(nvi_eager_resample, K=8, seed=1, eval_break=50, num_iterations=10000)

