import numpy as np

from expander_ldr.clustering import cluster_candidates, kmeans_pp_initialization


def test_kmeans_pp_initialization_returns_k_centers():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((10, 2))
    centers = kmeans_pp_initialization(points, k=3, rng=rng)
    assert centers.shape == (3,)
    assert len(set(centers.tolist())) == 3


def test_clustering_merges_nearby():
    rng = np.random.default_rng(0)
    cluster1 = rng.normal(loc=(0, 0), scale=0.1, size=(10, 2))
    cluster2 = rng.normal(loc=(10, 0), scale=0.1, size=(10, 2))
    cluster3 = rng.normal(loc=(0, 10), scale=0.1, size=(10, 2))
    candidates = [*cluster1, *cluster2, *cluster3]

    centers, labels = cluster_candidates(candidates, threshold=2.0)

    assert centers.shape == (3, 2)
    assert labels.shape == (len(candidates),)

    true_centers = np.array([[0, 0], [10, 0], [0, 10]])
    for center in centers:
        dists = np.linalg.norm(true_centers - center, axis=1)
        assert np.min(dists) < 0.5


def test_clustering_single_candidate():
    candidate = np.array([1.0, -1.0])
    centers, labels = cluster_candidates([candidate], threshold=1.0)

    assert centers.shape == (1, 2)
    assert labels.shape == (1,)
    assert np.allclose(centers[0], candidate)
    assert labels[0] == 0
