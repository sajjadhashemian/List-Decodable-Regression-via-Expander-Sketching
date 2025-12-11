"""Synthetic experiment interface and CLI helpers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .estimator import ExpanderLDRRegressor


@dataclass
class SyntheticConfig:
    """Configuration for a single synthetic list-decodable regression run."""

    n_train: int = 5000
    n_test: int = 2000
    d: int = 20
    alpha: float = 0.2
    noise_std: float = 0.1
    outlier_scale: float = 10.0
    seeds: int = 10
    random_state: int = 0
    repetitions: int = 8
    buckets: int = 200
    left_degree: int = 3
    filtering_rounds: int = 4
    blocks: int = 16
    ridge: float = 1e-3
    prune_eta: float = 0.2
    prune_rho: float = 0.1


class ExperimentRunner:
    """Run synthetic experiments and report metrics."""

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg

    def generate_data(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate training and test data for list-decodable regression."""

        cfg = self.cfg
        X_train = rng.standard_normal((cfg.n_train, cfg.d))
        X_test = rng.standard_normal((cfg.n_test, cfg.d))
        w_star = rng.standard_normal(cfg.d)

        noise_train = rng.normal(scale=cfg.noise_std, size=cfg.n_train)
        noise_test = rng.normal(scale=cfg.noise_std, size=cfg.n_test)

        y_in_train = X_train @ w_star + noise_train
        y_test = X_test @ w_star + noise_test

        n_inliers = int(cfg.alpha * cfg.n_train)
        perm = rng.permutation(cfg.n_train)
        inlier_idx = perm[:n_inliers]
        outlier_idx = perm[n_inliers:]

        y_train = np.zeros(cfg.n_train)
        y_train[inlier_idx] = y_in_train[inlier_idx]
        y_train[outlier_idx] = cfg.outlier_scale * rng.normal(size=len(outlier_idx))

        inlier_mask = np.zeros(cfg.n_train, dtype=bool)
        inlier_mask[inlier_idx] = True

        return X_train, y_train, X_test, y_test, w_star, inlier_mask

    def run(self) -> Dict[str, Any]:
        """Run a single experiment and report recovery/prediction errors."""

        rng = np.random.default_rng(self.cfg.random_state)
        X_train, y_train, X_test, y_test, w_star, inlier_mask = self.generate_data(rng)

        estimator = ExpanderLDRRegressor(
            alpha=self.cfg.alpha,
            repetitions=self.cfg.repetitions,
            buckets=self.cfg.buckets,
            left_degree=self.cfg.left_degree,
            filtering_rounds=self.cfg.filtering_rounds,
            seeds=self.cfg.seeds,
            blocks=self.cfg.blocks,
            ridge=self.cfg.ridge,
            prune_eta=self.cfg.prune_eta,
            prune_rho=self.cfg.prune_rho,
            random_state=self.cfg.random_state,
        )
        estimator.fit(X_train, y_train)

        recovery_error = float(
            np.min([np.linalg.norm(c - w_star) for c in estimator.candidates_])
        )
        test_mse = float(np.mean((estimator.predict(X_test) - X_test @ w_star) ** 2))

        return {
            "config": self.cfg,
            "recovery_error": recovery_error,
            "test_mse": test_mse,
            "n_inliers": int(inlier_mask.sum()),
        }

    def plot_results(self, results: Dict[str, Any]) -> None:
        """Plot recovery and prediction errors as a simple bar chart."""

        metrics = {"recovery_error": results["recovery_error"], "test_mse": results["test_mse"]}
        fig, ax = plt.subplots()
        ax.bar(list(metrics.keys()), list(metrics.values()))
        ax.set_ylabel("Error")
        ax.set_title("Expander LDR synthetic experiment")
        plt.tight_layout()
        plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic expander LDR experiment")
    parser.add_argument("--n-train", type=int, default=SyntheticConfig.n_train)
    parser.add_argument("--n-test", type=int, default=SyntheticConfig.n_test)
    parser.add_argument("--d", type=int, default=SyntheticConfig.d)
    parser.add_argument("--alpha", type=float, default=SyntheticConfig.alpha)
    parser.add_argument("--noise-std", type=float, default=SyntheticConfig.noise_std)
    parser.add_argument("--outlier-scale", type=float, default=SyntheticConfig.outlier_scale)
    parser.add_argument("--seeds", type=int, default=SyntheticConfig.seeds)
    parser.add_argument("--random-state", type=int, default=SyntheticConfig.random_state)
    return parser


def _cfg_from_args(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        alpha=args.alpha,
        noise_std=args.noise_std,
        outlier_scale=args.outlier_scale,
        seeds=args.seeds,
        random_state=args.random_state,
    )


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _cfg_from_args(args)
    runner = ExperimentRunner(cfg)
    results = runner.run()
    print(
        "recovery_error={:.4f}, test_mse={:.4f}, n_inliers={}".format(
            results["recovery_error"], results["test_mse"], results["n_inliers"]
        )
    )
    return results


if __name__ == "__main__":
    main()
