"""Run a synthetic expander LDR experiment from the command line.

Usage
-----
python examples/run_synthetic_experiment.py --n-train 5000 --d 20 --alpha 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from expander_ldr.experiments import ExperimentRunner, SyntheticConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic expander LDR experiment")
    parser.add_argument("--n-train", type=int, default=SyntheticConfig.n_train)
    parser.add_argument("--n-test", type=int, default=SyntheticConfig.n_test)
    parser.add_argument("--d", type=int, default=SyntheticConfig.d)
    parser.add_argument("--alpha", type=float, default=SyntheticConfig.alpha)
    parser.add_argument("--noise-std", type=float, default=SyntheticConfig.noise_std)
    parser.add_argument("--outlier-scale", type=float, default=SyntheticConfig.outlier_scale)
    parser.add_argument("--seeds", type=int, default=SyntheticConfig.seeds)
    parser.add_argument("--repetitions", type=int, default=SyntheticConfig.repetitions)
    parser.add_argument("--buckets", type=int, default=SyntheticConfig.buckets)
    parser.add_argument("--left-degree", type=int, default=SyntheticConfig.left_degree)
    parser.add_argument("--filtering-rounds", type=int, default=SyntheticConfig.filtering_rounds)
    parser.add_argument("--blocks", type=int, default=SyntheticConfig.blocks)
    parser.add_argument("--ridge", type=float, default=SyntheticConfig.ridge)
    parser.add_argument("--prune-eta", type=float, default=SyntheticConfig.prune_eta)
    parser.add_argument("--prune-rho", type=float, default=SyntheticConfig.prune_rho)
    parser.add_argument("--random-state", type=int, default=SyntheticConfig.random_state)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a simple bar plot of recovery and prediction errors",
    )
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
        repetitions=args.repetitions,
        buckets=args.buckets,
        left_degree=args.left_degree,
        filtering_rounds=args.filtering_rounds,
        blocks=args.blocks,
        ridge=args.ridge,
        prune_eta=args.prune_eta,
        prune_rho=args.prune_rho,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
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
    if args.plot:
        runner.plot_results(results)
    return results


if __name__ == "__main__":
    main()
