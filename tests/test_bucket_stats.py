import numpy as np

from expander_ldr.bucket_stats import BucketStatistics
from expander_ldr.expander import ExpanderSketcher


def test_bucket_moments_simple():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])

    bucket_indices = [[np.array([0, 1]), np.array([2, 3])]]
    bucket_signs = [[np.array([1.0, -1.0]), np.array([-1.0, 1.0])]]
    active_buckets = [(0, 0), (0, 1)]

    stats = BucketStatistics(X, y)
    moments = stats.compute_moments(bucket_indices, bucket_signs, active_buckets)

    H0, g0 = moments[0]
    X0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    y0 = np.array([1.0, -2.0])
    expected_H0 = (X0.T @ X0) / 2
    expected_g0 = (X0.T @ y0) / 2
    assert np.allclose(H0, expected_H0)
    assert np.allclose(g0, expected_g0)

    H1, g1 = moments[1]
    X1 = np.array([[-1.0, -1.0], [2.0, 0.0]])
    y1 = np.array([-3.0, 4.0])
    expected_H1 = (X1.T @ X1) / 2
    expected_g1 = (X1.T @ y1) / 2
    assert np.allclose(H1, expected_H1)
    assert np.allclose(g1, expected_g1)


def test_bucket_residual_covariance_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)

    sketcher = ExpanderSketcher(n_buckets=4, repetitions=2, left_degree=2, random_state=0)
    sketcher.fit(n_samples=X.shape[0])
    bucket_indices, bucket_signs = sketcher.get_bucket_assignment()

    active_buckets = [(t, b) for t in range(2) for b in range(4)]
    l_hat = rng.normal(size=5)

    stats = BucketStatistics(X, y)
    covariances = stats.compute_residual_covariances(
        l_hat, bucket_indices, bucket_signs, active_buckets
    )

    for C in covariances:
        assert C.shape == (5, 5)
        assert np.allclose(C, C.T)
