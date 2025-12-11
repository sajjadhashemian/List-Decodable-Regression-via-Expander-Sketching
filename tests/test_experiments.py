import subprocess
import sys
from pathlib import Path

import numpy as np

from expander_ldr.experiments import ExperimentRunner, SyntheticConfig


def test_generate_data_shapes():
    cfg = SyntheticConfig(n_train=100, n_test=50, d=5, alpha=0.3, random_state=1)
    runner = ExperimentRunner(cfg)
    rng = np.random.default_rng(cfg.random_state)
    X_train, y_train, X_test, y_test, w_star, inlier_mask = runner.generate_data(rng)

    assert X_train.shape == (cfg.n_train, cfg.d)
    assert y_train.shape == (cfg.n_train,)
    assert X_test.shape == (cfg.n_test, cfg.d)
    assert y_test.shape == (cfg.n_test,)
    assert w_star.shape == (cfg.d,)
    assert inlier_mask.shape == (cfg.n_train,)
    assert inlier_mask.sum() == int(cfg.alpha * cfg.n_train)


def test_experiment_runner_basic():
    cfg = SyntheticConfig(
        n_train=500,
        n_test=200,
        d=10,
        alpha=0.25,
        seeds=3,
        repetitions=3,
        buckets=60,
        left_degree=2,
        filtering_rounds=2,
        blocks=4,
        random_state=0,
    )
    runner = ExperimentRunner(cfg)
    results = runner.run()

    assert set(["recovery_error", "test_mse"]).issubset(results)
    assert np.isfinite(results["recovery_error"])
    assert np.isfinite(results["test_mse"])


def test_example_script_smoke():
    cmd = [
        sys.executable,
        "examples/run_synthetic_experiment.py",
        "--n-train",
        "200",
        "--n-test",
        "50",
        "--d",
        "5",
        "--alpha",
        "0.3",
        "--seeds",
        "1",
        "--repetitions",
        "2",
        "--buckets",
        "20",
        "--left-degree",
        "2",
        "--filtering-rounds",
        "1",
        "--blocks",
        "4",
        "--random-state",
        "0",
    ]
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
