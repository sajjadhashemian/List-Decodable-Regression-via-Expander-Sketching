import numpy as np

from expander_ldr.robust_agg import aggregate_moments, geometric_median, median_of_means


def test_median_of_means_recovers_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    mom = median_of_means(X, n_blocks=10)
    assert np.linalg.norm(mom) < 0.2


def test_geometric_median_simple():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    gm = geometric_median(X, max_iter=5000, tol=1e-8)

    grid = np.linspace(0, 1, 101)
    xv, yv = np.meshgrid(grid, grid)
    candidates = np.stack([xv.ravel(), yv.ravel()], axis=1)
    losses = np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2).sum(axis=1)
    best = candidates[losses.argmin()]

    assert np.linalg.norm(gm - best) < 0.1


def test_aggregate_moments_shapes():
    rng = np.random.default_rng(1)
    H_list = []
    g_list = []
    for _ in range(5):
        A = rng.normal(size=(5, 3))
        H = A.T @ A + 0.1 * np.eye(3)
        g = rng.normal(size=3)
        H_list.append(H)
        g_list.append(g)

    Sigma_hat_geom, g_hat_geom = aggregate_moments(H_list, g_list, method="geom_median")
    Sigma_hat_mom, g_hat_mom = aggregate_moments(H_list, g_list, method="mom", n_blocks=2)

    for Sigma_hat, g_hat in [(Sigma_hat_geom, g_hat_geom), (Sigma_hat_mom, g_hat_mom)]:
        assert Sigma_hat.shape == (3, 3)
        assert g_hat.shape == (3,)
        assert np.isfinite(Sigma_hat).all()
        assert np.isfinite(g_hat).all()
