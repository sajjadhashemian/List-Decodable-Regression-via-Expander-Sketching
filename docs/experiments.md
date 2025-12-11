# Experiments

The `expander_ldr.experiments` module provides a thin wrapper for running synthetic experiments and logging results.

## Synthetic list-decodable regression

The basic workflow:

1. Generate inliers \(X \in \mathbb{R}^{n \times d}\), \(y = X \ell_\star + \xi\).
2. Corrupt a \(1-\alpha\) fraction of samples arbitrarily.
3. Fit `ExpanderLDRRegressor`.
4. Measure:
   - Distance to ground truth: \(\min_k \|\hat\ell^{(k)} - \ell_\star\|_2\),
   - Prediction error on a fresh test set.

## ExperimentRunner

`ExperimentRunner` exposes:

- A declarative configuration (dataclass or dictionary),
- Methods to run a single trial or multiple repeats,
- Saving results (e.g. to JSON / NumPy files) and optional plots.

Example usage:

```python
from expander_ldr.experiments import ExperimentRunner, SyntheticConfig

cfg = SyntheticConfig(
    n_train=5000,
    n_test=2000,
    d=20,
    alpha=0.2,
    noise_std=0.1,
    outlier_scale=10.0,
    seeds=10,
)

runner = ExperimentRunner(cfg)
results = runner.run()
runner.plot_results(results)
```

See `examples/run_synthetic_experiment.py` for a concrete script.
