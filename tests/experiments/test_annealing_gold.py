#!/usr/bin/env python3

from experiments.annealing.main import *
from combinators.utils import git_root

def test_gold():
    seed = 1
    K = 8
    optimize_path = True
    objective = nvo_rkl
    resample = True
    S = 288
    iterations = 10

    torch.manual_seed(seed)
    model = mk_model(K, optimize_path=optimize_path)
    q = nvi_declarative(*model,
                        objective,
                        resample=resample)

    q, losses, ess, lZ_hat = nvi_train(
        q,
        *model,
        sample_shape=(S//K,1),
        iterations=iterations,
        batch_dim=1,
        sample_dims=0,
    )

    q = nvi_declarative(*model,
                        objective,
                        resample=False)

    losses_test, ess_test, lZ_hat_test, samples_test, _ = \
        nvi_test(q, (1000, 100), batch_dim=1, sample_dims=0)

    #torch.save((losses, ess, lZ_hat, losses_test, ess_test, lZ_hat_test, samples_test), 'test_annealing_gold.pt')

    losses_gold, ess_gold, lZ_hat_gold, losses_test_gold, ess_test_gold, lZ_hat_test_gold, samples_test_gold \
        = torch.load(git_root() + '/tests/experiments/test_annealing_gold.pt')
    assert torch.equal(losses_gold, losses)
    assert torch.equal(ess_gold, ess)
    assert torch.equal(lZ_hat_gold, lZ_hat)
    assert torch.equal(losses_test_gold, losses_test)
    assert torch.equal(ess_test_gold, ess_test)
    assert torch.equal(lZ_hat_test_gold, lZ_hat_test)

    for (kg, tensorg), (k, tensor) in zip(samples_test_gold, samples_test):
        assert kg == k
        assert torch.equal(tensorg, tensor)
