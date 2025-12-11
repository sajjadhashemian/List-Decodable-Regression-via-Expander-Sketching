"""Robust aggregation utilities such as median-of-means and geometric median."""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


def median_of_means(X: np.ndarray, n_blocks: int) -> np.ndarray:
    """Median-of-means estimator for a collection of vectors.

    Parameters
    ----------
    X : array, shape (m, d)
        Observations.
    n_blocks : int
        Number of blocks M.

    Returns
    -------
    mom : array, shape (d,)
        Coordinate-wise median of block means.
    """

    if X.ndim != 2:
        raise ValueError("median_of_means expects a 2D array")
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")

    blocks = np.array_split(X, n_blocks)
    block_means = np.stack([blk.mean(axis=0) for blk in blocks if blk.size], axis=0)
    return np.median(block_means, axis=0)


def geometric_median(X: np.ndarray, max_iter: int = 500, tol: float = 1e-6) -> np.ndarray:
    """Compute the geometric median of vectors using Weiszfeld's algorithm."""

    if X.ndim != 2:
        raise ValueError("geometric_median expects a 2D array")

    current = X.mean(axis=0)
    for _ in range(max_iter):
        diff = X - current
        distances = np.linalg.norm(diff, axis=1)
        # avoid division by zero
        weights = 1.0 / np.maximum(distances, 1e-12)
        next_point = (X * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(next_point - current) <= tol:
            current = next_point
            break
        current = next_point
    return current


def aggregate_moments(
    H_list: Sequence[np.ndarray],
    g_list: Sequence[np.ndarray],
    method: Literal["geom_median", "mom"] = "geom_median",
    n_blocks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate local (H, g) pairs into global (Sigma_hat, g_hat).

    - Flatten H matrices to vectors for MoM/geometric median, then reshape.
    - Use ``n_blocks`` for MoM when method="mom"; default to sqrt(m) blocks.
    """

    H_arr = np.asarray(H_list)
    g_arr = np.asarray(g_list)
    if H_arr.ndim != 3:
        raise ValueError("H_list must form an array of shape (m, d, d)")
    if g_arr.ndim != 2:
        raise ValueError("g_list must form an array of shape (m, d)")
    if H_arr.shape[0] != g_arr.shape[0]:
        raise ValueError("H_list and g_list must have the same length")

    m, d, _ = H_arr.shape
    if method == "mom":
        if n_blocks is None:
            n_blocks = int(np.sqrt(m)) or 1
        H_flat = H_arr.reshape(m, d * d)
        Sigma_hat = median_of_means(H_flat, n_blocks=n_blocks).reshape(d, d)
        g_hat = median_of_means(g_arr, n_blocks=n_blocks)
    elif method == "geom_median":
        H_flat = H_arr.reshape(m, d * d)
        Sigma_hat = geometric_median(H_flat).reshape(d, d)
        g_hat = geometric_median(g_arr)
    else:
        raise ValueError("method must be 'geom_median' or 'mom'")

    return Sigma_hat, g_hat
