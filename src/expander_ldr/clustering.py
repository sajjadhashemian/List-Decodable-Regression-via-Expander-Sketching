"""List generation via clustering candidate regressors."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Sequence, Tuple


def kmeans_pp_initialization(points: np.ndarray, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return indices of initial centers using a simple k-means++ style seeding."""

    if rng is None:
        rng = np.random.default_rng()

    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if k <= 0:
        raise ValueError("k must be positive")

    n_samples = points.shape[0]
    centers = [rng.integers(0, n_samples)]
    while len(centers) < k:
        dist_sq = np.min(np.linalg.norm(points - points[centers][:, None], axis=2) ** 2, axis=0)
        probs = dist_sq / dist_sq.sum()
        next_center = rng.choice(n_samples, p=probs)
        centers.append(int(next_center))
    return np.array(centers, dtype=int)


def cluster_candidates(
    candidates: Sequence[np.ndarray], threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster candidate regressors using single-linkage and an l2 threshold.

    Parameters
    ----------
    candidates : sequence of arrays, each shape (d,)
    threshold : float
        Distance threshold for single-linkage clustering.

    Returns
    -------
    centers : np.ndarray, shape (K, d)
        Cluster centers (chosen as medoids of each cluster).
    labels : np.ndarray, shape (len(candidates),)
        Cluster labels per candidate.
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive")

    if len(candidates) == 0:
        raise ValueError("candidates must be non-empty")

    if len(candidates) == 1:
        single = np.asarray(candidates[0], dtype=float)
        return single.reshape(1, -1), np.array([0], dtype=int)

    X = np.vstack([np.asarray(c, dtype=float) for c in candidates])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="single",
        metric="euclidean",
    )
    labels = clustering.fit_predict(X)

    centers = []
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = X[cluster_indices]
        # Compute total distance from each point to others in the cluster
        diffs = cluster_points[:, None, :] - cluster_points[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        total_dist = distances.sum(axis=1)
        medoid_idx = cluster_indices[np.argmin(total_dist)]
        centers.append(X[medoid_idx])

    centers_array = np.vstack(centers)
    return centers_array, labels
