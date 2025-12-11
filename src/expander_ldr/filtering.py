"""Spectral filtering loop on buckets."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .bucket_stats import BucketStatistics
from .robust_agg import aggregate_moments


class FilteringLoop:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bucket_indices,
        bucket_signs,
        alpha: float,
        repetitions: int,
        n_buckets: int,
        blocks: int,
        ridge: float,
        prune_eta: float,
        prune_rho: float,
        robust_method: str = "geom_median",
    ) -> None:
        self.stats = BucketStatistics(X, y)
        self.bucket_indices = bucket_indices
        self.bucket_signs = bucket_signs
        self.alpha = alpha
        self.repetitions = repetitions
        self.n_buckets = n_buckets
        self.blocks = blocks
        self.ridge = ridge
        self.prune_eta = prune_eta
        self.prune_rho = prune_rho
        self.robust_method = robust_method

        self.active_buckets: List[Tuple[int, int]] = [
            (t, b)
            for t in range(repetitions)
            for b in range(n_buckets)
            if bucket_indices[t][b].size > 0
        ]

    def _aggregate_covariances(self, cov_list: List[np.ndarray]) -> np.ndarray:
        if not cov_list:
            raise ValueError("No covariances to aggregate")
        d = cov_list[0].shape[0]
        zeros = [np.zeros(d) for _ in cov_list]
        cov_hat, _ = aggregate_moments(
            cov_list, zeros, method=self.robust_method, n_blocks=self.blocks
        )
        return cov_hat

    def run(
        self,
        max_rounds: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, dict]:
        """Run the filtering loop."""

        rng = np.random.default_rng(rng)
        info = {"lambda_max": [], "target_var": [], "n_active": []}

        active_buckets: List[Tuple[int, int]] = list(self.active_buckets)
        if not active_buckets:
            raise ValueError("No active buckets to process")

        for round_idx in range(max_rounds):
            info["n_active"].append(len(active_buckets))

            moments = self.stats.compute_moments(
                self.bucket_indices, self.bucket_signs, active_buckets
            )
            H_list, g_list = zip(*moments)
            Sigma_hat, g_hat = aggregate_moments(
                H_list,
                g_list,
                method=self.robust_method,
                n_blocks=self.blocks,
            )

            d = Sigma_hat.shape[0]
            Sigma_reg = Sigma_hat + self.ridge * np.eye(d)
            l_hat = np.linalg.solve(Sigma_reg, g_hat)

            if round_idx == max_rounds - 1:
                break

            covariances = self.stats.compute_residual_covariances(
                l_hat, self.bucket_indices, self.bucket_signs, active_buckets
            )
            C_hat = self._aggregate_covariances(covariances)

            evals, evecs = np.linalg.eigh(C_hat)
            lambda_max = float(evals[-1])
            target_var = float(np.median(evals))
            info["lambda_max"].append(lambda_max)
            info["target_var"].append(target_var)

            if lambda_max <= (1.0 + self.prune_eta) * target_var:
                break

            v = evecs[:, -1]
            scores = np.array([float(v.T @ C @ v) for C in covariances])
            n_drop = max(1, int(np.ceil(self.prune_rho * len(active_buckets))))
            drop_indices = np.argsort(scores)[-n_drop:]
            keep_mask = np.ones(len(active_buckets), dtype=bool)
            keep_mask[drop_indices] = False
            active_buckets = [b for b, keep in zip(active_buckets, keep_mask) if keep]

            if not active_buckets:
                break

        return l_hat, info
