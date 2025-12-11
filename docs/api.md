# API Reference

## `expander_ldr.ExpanderLDRRegressor`

```python
class ExpanderLDRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha: float,
        repetitions: int = 8,
        buckets: int = 200,
        left_degree: int = 3,
        filtering_rounds: int = 4,
        seeds: int = 10,
        blocks: int = 16,
        ridge: float = 0.0,
        prune_eta: float = 0.2,
        prune_rho: float = 0.1,
        clustering_threshold: float | None = None,
        robust_method: str = "geom_median",
        random_state: int | None = None,
        n_jobs: int | None = None,
    ): ...
```

**Attributes after fit:**
- `coef_`, `intercept_` – chosen candidate (e.g. smallest training error on inlier guess).
- `candidates_` – list of all candidate coef_ vectors (length ≤ seeds).
- `candidate_scores_` – scores (e.g. empirical loss) per candidate.
- `expander_params_` – actual parameters used internally.
- `n_features_in_` – as in scikit-learn.

**Methods:**
- `fit(X, y)` – run the full pipeline and store candidates.
- `predict(X)` – use the chosen candidate.
- `score(X, y)` – R² score, consistent with scikit-learn.
- `fit_predict(X, y)` – convenience wrapper returning predictions.
- `get_params`, `set_params` – for integration with scikit-learn tools.

## Lower-level modules

- `expander_ldr.expander.ExpanderSketcher` – build bucket assignments.
- `expander_ldr.bucket_stats.BucketStatistics` – per-bucket moments and residual covariances.
- `expander_ldr.robust_agg` – `median_of_means`, `geometric_median` utilities.
- `expander_ldr.filtering.FilteringLoop` – spectral filtering logic.
- `expander_ldr.clustering.cluster_candidates` – clustering for list generation.
- `expander_ldr.experiments.ExperimentRunner` – high-level experiment harness.
