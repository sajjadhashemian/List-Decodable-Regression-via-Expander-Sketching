"""Spectral filtering loop on buckets."""
from __future__ import annotations

from typing import List, Tuple
import time

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
        collect_diagnostics: bool = False,
        enable_filtering: bool = True,
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
        self.collect_diagnostics = collect_diagnostics
        self.enable_filtering = enable_filtering

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
        loop_start = time.perf_counter()
        info = {
            "lambda_max": [],
            "target_var": [],
            "n_active": [],
            "stopping_reason": "",
            "rounds_ran": 0,
            "aggregation_time": 0.0,
            "solve_time": 0.0,
            "filtering_time": 0.0,
        }
        if self.collect_diagnostics:
            info.update(
                {
                    "active_buckets_by_round": [],
                    "pruned_buckets_by_round": [],
                    "bucket_scores_by_round": [],
                    "bucket_score_histograms": [],
                }
            )

        active_buckets: List[Tuple[int, int]] = list(self.active_buckets)
        if not active_buckets:
            raise ValueError("No active buckets to process")

        if max_rounds <= 0 or not self.enable_filtering:
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
            info["stopping_reason"] = "filtering_disabled"
            info["rounds_ran"] = 1
            info["n_active"].append(len(active_buckets))
            info["filtering_time"] = float(time.perf_counter() - loop_start)
            return l_hat, info

        for round_idx in range(max_rounds):
            info["n_active"].append(len(active_buckets))

            if self.collect_diagnostics:
                info["active_buckets_by_round"].append(list(active_buckets))

            moments = self.stats.compute_moments(
                self.bucket_indices, self.bucket_signs, active_buckets
            )
            H_list, g_list = zip(*moments)
            aggregation_start = time.perf_counter()
            Sigma_hat, g_hat = aggregate_moments(
                H_list,
                g_list,
                method=self.robust_method,
                n_blocks=self.blocks,
            )
            info["aggregation_time"] += float(time.perf_counter() - aggregation_start)
            solve_start = time.perf_counter()

            d = Sigma_hat.shape[0]
            Sigma_reg = Sigma_hat + self.ridge * np.eye(d)
            l_hat = np.linalg.solve(Sigma_reg, g_hat)
            info["solve_time"] += float(time.perf_counter() - solve_start)

            if round_idx == max_rounds - 1:
                if self.collect_diagnostics:
                    info["bucket_scores_by_round"].append(np.array([]))
                    info["bucket_score_histograms"].append((np.array([]), np.array([])))
                    info["pruned_buckets_by_round"].append([])
                info["stopping_reason"] = "max_rounds"
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

            v = evecs[:, -1]
            scores = np.array([float(v.T @ C @ v) for C in covariances])
            if self.collect_diagnostics:
                info["bucket_scores_by_round"].append(scores)
                hist = np.histogram(scores, bins=20)
                info["bucket_score_histograms"].append(hist)

            if lambda_max <= (1.0 + self.prune_eta) * target_var:
                if self.collect_diagnostics:
                    info["pruned_buckets_by_round"].append([])
                info["stopping_reason"] = "spectral_ok"
                break

            n_drop = max(1, int(np.ceil(self.prune_rho * len(active_buckets))))
            drop_indices = np.argsort(scores)[-n_drop:]
            keep_mask = np.ones(len(active_buckets), dtype=bool)
            keep_mask[drop_indices] = False
            if self.collect_diagnostics:
                pruned = [active_buckets[i] for i in drop_indices]
                info["pruned_buckets_by_round"].append(pruned)
            active_buckets = [b for b, keep in zip(active_buckets, keep_mask) if keep]

            if not active_buckets:
                info["stopping_reason"] = "no_active_buckets"
                break

            info["rounds_ran"] = round_idx + 1

        if not info["stopping_reason"]:
            info["stopping_reason"] = "completed"
        info["rounds_ran"] = round_idx + 1

        info["filtering_time"] = float(time.perf_counter() - loop_start)
        return l_hat, info
