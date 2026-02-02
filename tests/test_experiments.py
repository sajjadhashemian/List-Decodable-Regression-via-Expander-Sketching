from pathlib import Path

from experiments.runner import ExperimentConfig, SyntheticConfig, run_config


def test_experiment_runner_smoke(tmp_path):
    cfg = SyntheticConfig(
        n_train=300,
        n_test=100,
        d=10,
        alpha=0.25,
        seeds=3,
        repetitions=2,
        buckets=30,
        left_degree=2,
        filtering_rounds=1,
        blocks=2,
        random_seed=0,
        n_trials=1,
        outdir=str(tmp_path),
    )
    exp_cfg = ExperimentConfig(
        name="smoke",
        variant="tiny",
        data_type="synthetic",
        config=cfg,
    )
    rows = run_config(exp_cfg, outdir=str(tmp_path))
    assert rows
    row = rows[0]
    assert "oracle_param_error" in row
    assert Path(tmp_path, "results.jsonl").exists()
    assert Path(tmp_path, "summary.csv").exists()
    assert Path(tmp_path, "plots", "smoke_oracle_error_hist.png").exists()
