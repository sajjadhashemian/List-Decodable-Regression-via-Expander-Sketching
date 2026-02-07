# Code Reference (Python Modules)

This document provides a file-by-file walkthrough of the Python code in the
repository, describing each module's role, major classes/functions, and how
it contributes to the list-decodable regression pipeline.

## Package modules (`src/expander_ldr`)

### `__init__.py`
- Defines the package interface and re-exports the primary estimator
  `ExpanderLDRRegressor` for convenient imports.

### `bucket_stats.py`
- Implements `BucketStatistics`, the low-level numerical engine for computing
  per-bucket statistics.
- Responsibilities:
  - Computes bucket-wise moment pairs `(H, g)` given bucket assignments and
    sketch signs.
  - Computes residual covariance matrices for filtering rounds.
- Used by the filtering loop to build robust aggregated normal equations.

### `clustering.py`
- Implements candidate clustering utilities.
- Main entry point is `cluster_candidates`, which performs single-linkage
  clustering on candidate regressors using a distance threshold.
- Includes `kmeans_pp_initialization` for optional seeding strategies.
- Used by the estimator to merge per-seed candidates into a final list.

### `diagnostics.py`
- Provides diagnostic utilities for mechanistic experiments (E5 series).
- `bucket_contamination_stats` summarizes inlier/outlier counts per bucket and
  computes contamination fractions plus unique-neighbor inlier statistics.
- `pruning_precision_recall` computes precision/recall curves for bucket
  pruning based on ground-truth contamination fractions.

### `estimator.py`
- Defines the `ExpanderLDRRegressor`, the scikit-learn compatible estimator.
- Responsibilities:
  - Orchestrates sketcher selection (expander/one-hash/countsketch/fixed).
  - Runs `FilteringLoop` per seed and aggregates candidate regressors.
  - Clusters candidates, caps list size, and provides prediction helpers.
  - Records diagnostics (`diagnostics_`) and timing breakdowns (`timings_`).
- Methods:
  - `fit`: runs the full list-decodable regression pipeline.
  - `predict`: predicts using the first candidate (backward compatible).
  - `predict_all`: returns predictions for all candidates.

### `expander.py`
- Implements expander sketch construction (`ExpanderSketcher`).
- Key responsibilities:
  - Samples left-regular bipartite graphs with configurable left degree.
  - Supports optional replacement and optional Rademacher signs.
  - Provides bucket assignments for each repetition via
    `get_bucket_assignment`.
- Utility functions `_as_rng`, `_sample_distinct_neighbors`, and
  `_invert_to_buckets` support randomization and adjacency inversion.

### `experiments.py`
- A lightweight wrapper that preserves the legacy experiment entry point.
- Delegates to the full runner in `experiments/runner.py` by constructing an
  `ExperimentConfig` and calling `run_config`.

### `filtering.py`
- Implements the spectral filtering loop (`FilteringLoop`).
- Responsibilities:
  - Aggregates moments via robust aggregation (mean/MoM/geom median).
  - Solves regularized normal equations for candidate regressors.
  - Performs residual-based spectral filtering with pruning.
  - Emits rich diagnostics (round-level metrics, bucket scores, histograms).
- Supports disabling filtering while keeping moment aggregation.

### `robust_agg.py`
- Robust aggregation utilities for matrix/vector moments.
- `median_of_means` and `geometric_median` implement standard estimators.
- `aggregate_moments` combines bucket-wise `(H, g)` using
  `geom_median`, `mom`, or `mean`.

### `run_experiments.py`
- CLI wrapper that exposes the experiment runner as
  `python -m expander_ldr.run_experiments`.
- Delegates to `experiments.runner.main`.

### `sketchers.py`
- Defines sketcher abstractions beyond the expander:
  - `OneHashSketcher`: one bucket per sample per repetition.
  - `CountSketchSketcher`: CountSketch-style hashing with optional signs.
  - `FixedBucketSketcher`: returns pre-specified bucket assignments (used for
    true-batch experiments).
- Provides a `BaseSketcher` protocol for typing and interchangeability.

### `utils.py`
- Miscellaneous helpers for shape checks and synthetic generation.
- Implements `select_by_median_squared_residual` for practical list selection
  and deterministic list capping.

## Experiment harness (`experiments`)

### `experiments/__init__.py`
- Re-exports experiment configuration classes and runner functions for
  convenience.

### `experiments/runner.py`
- The full experiment suite covering E1A–E8B.
- Responsibilities:
  - Defines run configs for synthetic and real-covariate experiments.
  - Implements synthetic data generation, corruption models, and batched data.
  - Implements real-covariate conversion protocol (ridge-based \(\ell^\star\)).
  - Runs baselines (CountSketch+OLS, JL+OLS, Huber, RANSAC, etc.).
  - Computes all required metrics (oracle error/risk, selection risk, rank
    curves, top-k success, timings, diagnostics).
  - Writes JSONL trial records, summary CSVs, and plots.
  - Maintains experiment registry mapping names to config grids.

## Examples (`examples`)

### `examples/run_synthetic_experiment.py`
- CLI helper for running a single synthetic configuration using the shared
  experiment runner.
- Useful for quick sanity checks without the full experiment registry.

## Tests (`tests`)

### `tests/conftest.py`
- Pytest configuration shared across the test suite.

### `tests/test_bucket_stats.py`
- Validates `BucketStatistics` moment and covariance computations.

### `tests/test_clustering.py`
- Verifies clustering logic and candidate grouping behavior.

### `tests/test_estimator_api.py`
- Ensures estimator fit/predict round-trips and list-size capping are correct.

### `tests/test_expander.py`
- Checks expander sketch reproducibility and sign behavior.

### `tests/test_experiments.py`
- Smoke test for the experiment runner ensuring JSON/CSV/plot outputs.

### `tests/test_filtering.py`
- Validates filtering convergence behavior and diagnostics lengths.

### `tests/test_robust_agg.py`
- Tests robust aggregation methods and output shapes.

### `tests/test_selection_utils.py`
- Confirms median-residual selection returns the correct candidate.

### `tests/test_sketchers.py`
- Ensures one-hash and fixed-bucket sketchers produce correct assignments.
