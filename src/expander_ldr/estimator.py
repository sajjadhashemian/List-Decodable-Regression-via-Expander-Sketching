"""Scikit-learn compatible estimator for expander-sketched list-decodable regression."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

from .clustering import cluster_candidates
from .expander import ExpanderSketcher
from .filtering import FilteringLoop


class ExpanderLDRRegressor(BaseEstimator, RegressorMixin):
    """Expander-sketched list-decodable linear regressor.

    Parameters
    ----------
    alpha : float
        Inlier fraction (between 0 and 0.5].
    repetitions : int, default=8
        Number of expander repetitions (``r`` in the algorithm).
    buckets : int, default=200
        Number of buckets per repetition (``B`` in the algorithm).
    left_degree : int, default=3
        Left degree of the expander graph (number of buckets per sample).
    filtering_rounds : int, default=4
        Maximum number of spectral filtering rounds.
    seeds : int, default=10
        Number of random seeds / repetitions of the full pipeline.
    blocks : int, default=16
        Number of blocks for robust aggregation.
    ridge : float, default=0.0
        Ridge regularization when solving the sketched normal equations.
    prune_eta : float, default=0.2
        Tolerance for eigenvalue growth before pruning.
    prune_rho : float, default=0.1
        Fraction of buckets to prune when eigenvalue test fails.
    clustering_threshold : float or None, default=None
        Distance threshold for single-linkage clustering of candidates. When
        ``None``, a heuristic based on the median distance between candidates
        is used.
    robust_method : {"geom_median", "mom"}, default="geom_median"
        Robust aggregation method for moments and residual covariances.
    random_state : int or None, default=None
        Seed for reproducibility.
    n_jobs : int or None, default=None
        Present for scikit-learn compatibility; unused in this implementation.
    """

    def __init__(
        self,
        alpha: float,
        repetitions: int = 8,
        buckets: int = 200,
        left_degree: int = 3,
        filtering_rounds: int = 4,
        seeds: int = 10,
        blocks: int = 16,
        ridge: float = 0.0,
        prune_eta: float = 0.2,
        prune_rho: float = 0.1,
        clustering_threshold: Optional[float] = None,
        robust_method: str = "geom_median",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.alpha = alpha
        self.repetitions = repetitions
        self.buckets = buckets
        self.left_degree = left_degree
        self.filtering_rounds = filtering_rounds
        self.seeds = seeds
        self.blocks = blocks
        self.ridge = ridge
        self.prune_eta = prune_eta
        self.prune_rho = prune_rho
        self.clustering_threshold = clustering_threshold
        self.robust_method = robust_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the list-decodable regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """

        X, y = check_X_y(X, y, dtype=np.float64, ensure_2d=True)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)
        candidate_list: List[np.ndarray] = []

        for _ in range(self.seeds):
            seed_value = rng.integers(np.iinfo(np.int32).max)
            expander = ExpanderSketcher(
                n_buckets=self.buckets,
                repetitions=self.repetitions,
                left_degree=self.left_degree,
                random_state=seed_value,
            )
            expander.fit(n_samples=n_samples)
            bucket_indices, bucket_signs = expander.get_bucket_assignment()

            loop_rng = np.random.default_rng(seed_value)
            filtering = FilteringLoop(
                X,
                y,
                bucket_indices,
                bucket_signs,
                alpha=self.alpha,
                repetitions=self.repetitions,
                n_buckets=self.buckets,
                blocks=self.blocks,
                ridge=self.ridge,
                prune_eta=self.prune_eta,
                prune_rho=self.prune_rho,
                robust_method=self.robust_method,
            )

            l_hat, _ = filtering.run(
                max_rounds=self.filtering_rounds, rng=loop_rng
            )
            candidate_list.append(l_hat)

        candidates = np.vstack(candidate_list)

        threshold = self.clustering_threshold
        if threshold is None:
            median_candidate = np.median(candidates, axis=0)
            distances = np.linalg.norm(candidates - median_candidate, axis=1)
            median_distance = np.median(distances)
            threshold = 2.0 * median_distance if median_distance > 0 else 1.0

        centers, labels = cluster_candidates(candidates, threshold=threshold)

        self.candidates_ = centers
        self.candidate_labels_ = labels
        self.coef_ = centers[0]
        self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses for the provided design matrix."""

        if not hasattr(self, "coef_"):
            raise ValueError("This ExpanderLDRRegressor instance is not fitted yet.")

        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        return X_checked @ self.coef_ + self.intercept_

