"""Experiment harness wrapper for expander LDR experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from experiments.runner import ExperimentConfig, SyntheticConfig, run_config


@dataclass
class ExperimentRunner:
    """Backward-compatible wrapper for running a single synthetic experiment."""

    cfg: Optional[SyntheticConfig] = None

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = SyntheticConfig()

    def run(self) -> Dict[str, Any]:
        exp_cfg = ExperimentConfig(
            name="custom",
            variant="synthetic",
            data_type="synthetic",
            config=self.cfg,
        )
        rows = run_config(exp_cfg, outdir=self.cfg.outdir, n_trials=self.cfg.n_trials)
        return rows[0]
