"""Scikit-learn compatible estimator for expander-sketched list-decodable regression."""
from __future__ import annotations

from typing import List, Optional

import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

from .clustering import cluster_candidates
from .expander import ExpanderSketcher
from .filtering import FilteringLoop
from .sketchers import CountSketchSketcher, FixedBucketSketcher, OneHashSketcher
from .utils import select_by_median_squared_residual


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
    robust_method : {"geom_median", "mom", "mean"}, default="geom_median"
        Robust aggregation method for moments and residual covariances.
    sketch_type : {"expander", "one_hash", "countsketch", "fixed"}, default="expander"
        Sketching strategy for bucket assignment.
    allow_replacement : bool, default=False
        Allow repeated buckets per sample when using expander sketching.
    use_signs : bool, default=True
        Use Rademacher signs in the sketch; if False, all signs are +1.
    list_size_cap : int or None, default=None
        Maximum number of candidates to keep after clustering. Defaults to ceil(8/alpha).
    return_diagnostics : bool, default=False
        If True, store per-seed diagnostics from filtering and sketching.
    timing : bool, default=True
        If True, store timing breakdowns in ``timings_``.
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
        sketch_type: str = "expander",
        allow_replacement: bool = False,
        use_signs: bool = True,
        list_size_cap: Optional[int] = None,
        return_diagnostics: bool = False,
        timing: bool = True,
        fixed_bucket_indices: Optional[List[List[np.ndarray]]] = None,
        fixed_bucket_signs: Optional[List[List[np.ndarray]]] = None,
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
        self.sketch_type = sketch_type
        self.allow_replacement = allow_replacement
        self.use_signs = use_signs
        self.list_size_cap = list_size_cap
        self.return_diagnostics = return_diagnostics
        self.timing = timing
        self.fixed_bucket_indices = fixed_bucket_indices
        self.fixed_bucket_signs = fixed_bucket_signs
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
        diagnostics = []
        timings = {
            "fit_total": 0.0,
            "sketch": 0.0,
            "filtering": 0.0,
            "clustering": 0.0,
            "aggregation": 0.0,
            "solve": 0.0,
        }
        fit_start = time.perf_counter()

        for _ in range(self.seeds):
            seed_value = rng.integers(np.iinfo(np.int32).max)
            if self.sketch_type == "expander":
                sketcher = ExpanderSketcher(
                    n_buckets=self.buckets,
                    repetitions=self.repetitions,
                    left_degree=self.left_degree,
                    random_state=seed_value,
                    allow_replacement=self.allow_replacement,
                    use_signs=self.use_signs,
                )
            elif self.sketch_type == "one_hash":
                sketcher = OneHashSketcher(
                    n_buckets=self.buckets,
                    repetitions=self.repetitions,
                    random_state=seed_value,
                    use_signs=self.use_signs,
                )
            elif self.sketch_type == "countsketch":
                sketcher = CountSketchSketcher(
                    n_buckets=self.buckets,
                    repetitions=self.repetitions,
                    random_state=seed_value,
                    use_signs=self.use_signs,
                )
            elif self.sketch_type == "fixed":
                if self.fixed_bucket_indices is None or self.fixed_bucket_signs is None:
                    raise ValueError(
                        "fixed sketch_type requires fixed_bucket_indices and fixed_bucket_signs."
                    )
                sketcher = FixedBucketSketcher(
                    bucket_indices=self.fixed_bucket_indices,
                    bucket_signs=self.fixed_bucket_signs,
                )
            else:
                raise ValueError(
                    "sketch_type must be one of 'expander', 'one_hash', 'countsketch', or 'fixed'"
                )

            sketch_start = time.perf_counter()
            sketcher.fit(n_samples=n_samples)
            bucket_indices, bucket_signs = sketcher.get_bucket_assignment()
            if self.timing:
                timings["sketch"] += float(time.perf_counter() - sketch_start)

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
                collect_diagnostics=self.return_diagnostics,
            )

            filtering_start = time.perf_counter()
            l_hat, info = filtering.run(
                max_rounds=self.filtering_rounds, rng=loop_rng
            )
            if self.timing:
                timings["filtering"] += float(time.perf_counter() - filtering_start)
                timings["aggregation"] += float(info.get("aggregation_time", 0.0))
                timings["solve"] += float(info.get("solve_time", 0.0))
            candidate_list.append(l_hat)
            if self.return_diagnostics:
                bucket_sizes = np.array(
                    [
                        bucket_indices[t][b].size
                        for t in range(len(bucket_indices))
                        for b in range(len(bucket_indices[t]))
                    ],
                    dtype=int,
                )
                diagnostics.append(
                    {
                        "filtering_info": info,
                        "bucket_sizes": bucket_sizes,
                        "sketch_type": self.sketch_type,
                        "bucket_indices": bucket_indices,
                        "bucket_signs": bucket_signs,
                    }
                )

        candidates = np.vstack(candidate_list)

        threshold = self.clustering_threshold
        if threshold is None:
            median_candidate = np.median(candidates, axis=0)
            distances = np.linalg.norm(candidates - median_candidate, axis=1)
            median_distance = np.median(distances)
            threshold = 2.0 * median_distance if median_distance > 0 else 1.0

        clustering_start = time.perf_counter()
        centers, labels = cluster_candidates(candidates, threshold=threshold)
        if self.timing:
            timings["clustering"] += float(time.perf_counter() - clustering_start)

        cap = self.list_size_cap
        if cap is None:
            cap = int(np.ceil(8.0 / self.alpha))
        if cap < centers.shape[0]:
            _, _, scores = select_by_median_squared_residual(X, y, centers)
            keep = np.argsort(scores)[:cap]
            centers_capped = centers[keep]
        else:
            centers_capped = centers

        self.candidates_raw_ = candidates
        self.candidates_uncapped_ = centers
        self.candidate_labels_ = labels
        self.candidates_ = centers_capped
        self.coef_ = centers_capped[0]
        self.intercept_ = 0.0
        if self.return_diagnostics:
            self.diagnostics_ = diagnostics
        if self.timing:
            timings["fit_total"] = float(time.perf_counter() - fit_start)
            self.timings_ = timings
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict responses for the provided design matrix."""

        if not hasattr(self, "coef_"):
            raise ValueError("This ExpanderLDRRegressor instance is not fitted yet.")

        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        return X_checked @ self.coef_ + self.intercept_

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "candidates_"):
            raise ValueError("This ExpanderLDRRegressor instance is not fitted yet.")
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        return self.candidates_ @ X_checked.T
