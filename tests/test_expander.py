import numpy as np

from expander_ldr.expander import ExpanderSketcher


def test_expander_shapes():
    sketcher = ExpanderSketcher(
        n_buckets=10, repetitions=3, left_degree=2, random_state=0
    )
    sketcher.fit(n_samples=100)

    assert len(sketcher.bucket_indices_) == 3
    assert len(sketcher.bucket_signs_) == 3

    total_assignments = 0
    for t in range(sketcher.repetitions):
        for b in range(sketcher.n_buckets):
            indices = sketcher.bucket_indices_[t][b]
            signs = sketcher.bucket_signs_[t][b]
            assert indices.shape == signs.shape
            assert set(np.unique(signs)).issubset({-1.0, 1.0})
            total_assignments += len(indices)

    assert total_assignments == 100 * sketcher.repetitions * sketcher.left_degree


def test_expander_reproducible():
    sketcher1 = ExpanderSketcher(
        n_buckets=8, repetitions=4, left_degree=3, random_state=42
    ).fit(n_samples=200)
    sketcher2 = ExpanderSketcher(
        n_buckets=8, repetitions=4, left_degree=3, random_state=42
    ).fit(n_samples=200)

    for t in range(sketcher1.repetitions):
        for b in range(sketcher1.n_buckets):
            np.testing.assert_array_equal(
                sketcher1.bucket_indices_[t][b], sketcher2.bucket_indices_[t][b]
            )
            np.testing.assert_array_equal(
                sketcher1.bucket_signs_[t][b], sketcher2.bucket_signs_[t][b]
            )


def test_nonregular_right_degrees():
    sketcher = ExpanderSketcher(
        n_buckets=7,
        repetitions=1,
        left_degree=3,
        random_state=1,
        regularity="configuration_model",
    ).fit(n_samples=5)

    total_assignments = sum(len(bucket) for bucket in sketcher.bucket_indices_[0])
    assert total_assignments == 5 * sketcher.left_degree


def test_near_regular_distribution_when_not_divisible():
    sketcher = ExpanderSketcher(
        n_buckets=7, repetitions=1, left_degree=3, random_state=0, regularity="regular"
    ).fit(n_samples=5)

    bucket_sizes = [len(bucket) for bucket in sketcher.bucket_indices_[0]]
    assert max(bucket_sizes) - min(bucket_sizes) <= 1
