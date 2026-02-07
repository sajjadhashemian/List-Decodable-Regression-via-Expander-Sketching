"""Shared helpers for checks, RNG, and synthetic data generation."""
from __future__ import annotations

import numpy as np


def check_dimensions(X: np.ndarray, y: np.ndarray) -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X must match length of y")


def make_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple synthetic regression problem."""

    if rng is None:
        rng = np.random.default_rng()
    X = rng.standard_normal((n_samples, n_features))
    coef = rng.standard_normal(n_features)
    y = X @ coef + rng.normal(scale=noise, size=n_samples)
    return X, y, coef


def select_by_median_squared_residual(
    X: np.ndarray, y: np.ndarray, candidates: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2D array")
    residuals = y[None, :] - candidates @ X.T
    scores = np.median(residuals**2, axis=1)
    best_idx = int(np.argmin(scores))
    return best_idx, candidates[best_idx], scores
