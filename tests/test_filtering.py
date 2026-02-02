import numpy as np

from expander_ldr.expander import ExpanderSketcher
from expander_ldr.filtering import FilteringLoop


def test_filtering_converges_trivially():
    rng = np.random.default_rng(42)
    n, d = 200, 5
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true

    sketcher = ExpanderSketcher(n_buckets=20, repetitions=3, left_degree=2, random_state=0)
    sketcher.fit(n_samples=n)
    bucket_indices, bucket_signs = sketcher.get_bucket_assignment()

    loop = FilteringLoop(
        X,
        y,
        bucket_indices,
        bucket_signs,
        alpha=0.5,
        repetitions=3,
        n_buckets=20,
        blocks=4,
        ridge=0.0,
        prune_eta=0.5,
        prune_rho=0.2,
    )

    max_rounds = 5
    _, info = loop.run(max_rounds=max_rounds)
    assert len(info["lambda_max"]) < max_rounds


def test_filtering_output_shape():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + 0.1 * rng.normal(size=n)

    sketcher = ExpanderSketcher(n_buckets=30, repetitions=4, left_degree=3, random_state=1)
    sketcher.fit(n_samples=n)
    bucket_indices, bucket_signs = sketcher.get_bucket_assignment()

    loop = FilteringLoop(
        X,
        y,
        bucket_indices,
        bucket_signs,
        alpha=0.4,
        repetitions=4,
        n_buckets=30,
        blocks=5,
        ridge=1e-3,
        prune_eta=0.2,
        prune_rho=0.2,
    )

    l_hat, info = loop.run(max_rounds=4)
    assert l_hat.shape == (d,)

    n_active = info["n_active"]
    assert all(x > 0 for x in n_active)
    assert all(n_active[i] >= n_active[i + 1] for i in range(len(n_active) - 1))


def test_filtering_diagnostics_lengths():
    rng = np.random.default_rng(1)
    n, d = 120, 4
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d)

    sketcher = ExpanderSketcher(n_buckets=10, repetitions=2, left_degree=2, random_state=2)
    sketcher.fit(n_samples=n)
    bucket_indices, bucket_signs = sketcher.get_bucket_assignment()

    loop = FilteringLoop(
        X,
        y,
        bucket_indices,
        bucket_signs,
        alpha=0.5,
        repetitions=2,
        n_buckets=10,
        blocks=2,
        ridge=0.0,
        prune_eta=0.2,
        prune_rho=0.5,
        collect_diagnostics=True,
    )
    _, info = loop.run(max_rounds=3)
    assert len(info["bucket_scores_by_round"]) == info["rounds_ran"]
    assert len(info["pruned_buckets_by_round"]) == info["rounds_ran"]
