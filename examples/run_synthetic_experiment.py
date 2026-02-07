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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.runner import ExperimentConfig, SyntheticConfig, run_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synthetic expander LDR experiment"
    )
    parser.add_argument("--n-train", type=int, default=SyntheticConfig.n_train)
    parser.add_argument("--n-test", type=int, default=SyntheticConfig.n_test)
    parser.add_argument("--d", type=int, default=SyntheticConfig.d)
    parser.add_argument("--alpha", type=float, default=SyntheticConfig.alpha)
    parser.add_argument("--noise-sigma", type=float, default=SyntheticConfig.noise_sigma)
    parser.add_argument("--seeds", type=int, default=SyntheticConfig.seeds)
    parser.add_argument("--repetitions", type=int, default=SyntheticConfig.repetitions)
    parser.add_argument("--buckets", type=int, default=SyntheticConfig.buckets)
    parser.add_argument("--left-degree", type=int, default=SyntheticConfig.left_degree)
    parser.add_argument(
        "--filtering-rounds", type=int, default=SyntheticConfig.filtering_rounds
    )
    parser.add_argument("--blocks", type=int, default=SyntheticConfig.blocks)
    parser.add_argument("--ridge", type=float, default=SyntheticConfig.ridge)
    parser.add_argument("--prune-eta", type=float, default=SyntheticConfig.prune_eta)
    parser.add_argument("--prune-rho", type=float, default=SyntheticConfig.prune_rho)
    parser.add_argument("--random-seed", type=int, default=SyntheticConfig.random_seed)
    parser.add_argument("--outdir", type=str, default="results/synthetic")
    return parser


def _cfg_from_args(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        alpha=args.alpha,
        noise_sigma=args.noise_sigma,
        seeds=args.seeds,
        repetitions=args.repetitions,
        buckets=args.buckets,
        left_degree=args.left_degree,
        filtering_rounds=args.filtering_rounds,
        blocks=args.blocks,
        ridge=args.ridge,
        prune_eta=args.prune_eta,
        prune_rho=args.prune_rho,
        random_seed=args.random_seed,
        outdir=args.outdir,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _cfg_from_args(args)
    exp_cfg = ExperimentConfig(
        name="custom",
        variant="synthetic",
        data_type="synthetic",
        config=cfg,
    )
    rows = run_config(exp_cfg, outdir=args.outdir)
    row = rows[0]
    print(
        "oracle_param_error={:.4f}, oracle_pred_risk={:.4f}, list_size={}".format(
            row["oracle_param_error"], row["oracle_pred_risk"], row["list_size"]
        )
    )
    return row


if __name__ == "__main__":
    main()
