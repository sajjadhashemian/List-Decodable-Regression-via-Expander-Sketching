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
