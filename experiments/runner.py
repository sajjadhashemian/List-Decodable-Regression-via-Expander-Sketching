from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml, load_svmlight_file
from sklearn.linear_model import (
    HuberRegressor,
    QuantileRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)

from expander_ldr.diagnostics import (
    bucket_contamination_stats,
    pruning_precision_recall,
)
from expander_ldr.estimator import ExpanderLDRRegressor
from expander_ldr.utils import select_by_median_squared_residual


@dataclass
class BaseRunConfig:
    random_seed: int = 0
    n_trials: int = 1
    outdir: str = "results"
    save_raw: bool = True


@dataclass
class SyntheticConfig(BaseRunConfig):
    n_train: int = 5000
    n_test: int = 2000
    d: int = 20
    alpha: float = 0.2
    noise_sigma: float = 0.1
    corruption_model: str = "label_only"
    leverage_tau: float = 10.0
    heavy_tail_nu: int = 5
    repetitions: int = 8
    buckets: int = 200
    left_degree: int = 3
    filtering_rounds: int = 4
    seeds: int = 10
    blocks: int = 16
    ridge: float = 1e-3
    prune_eta: float = 0.2
    prune_rho: float = 0.1
    robust_method: str = "geom_median"
    sketch_type: str = "expander"
    allow_replacement: bool = False
    use_signs: bool = True
    list_size_cap: Optional[int] = None


@dataclass
class RealDataConfig(BaseRunConfig):
    dataset_name: str = "YearPredictionMSD"
    n_subsample: Optional[int] = None
    alpha: float = 0.2
    corruption_model: str = "label_only"
    noise_sigma: float = 0.05
    random_seed: int = 0
    repetitions: int = 8
    buckets: int = 200
    left_degree: int = 3
    filtering_rounds: int = 4
    seeds: int = 10
    blocks: int = 16
    ridge: float = 1e-3
    prune_eta: float = 0.2
    prune_rho: float = 0.1
    robust_method: str = "geom_median"
    sketch_type: str = "expander"
    allow_replacement: bool = False
    use_signs: bool = True
    list_size_cap: Optional[int] = None
    foreign_dataset: Optional[str] = None
    libsvm_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    name: str
    variant: str
    data_type: str
    config: BaseRunConfig
    baselines: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


BASELINES = {
    "huber": HuberRegressor,
    "theilsen": TheilSenRegressor,
    "ransac": RANSACRegressor,
    "lad": QuantileRegressor,
}


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std, mean, std


def _generate_gaussian_data(cfg: SyntheticConfig, rng: np.random.Generator):
    X_train = rng.standard_normal((cfg.n_train, cfg.d))
    X_test = rng.standard_normal((cfg.n_test, cfg.d))
    w_star = rng.standard_normal(cfg.d)
    w_star = w_star / np.linalg.norm(w_star)
    noise_train = rng.normal(scale=cfg.noise_sigma, size=cfg.n_train)
    noise_test = rng.normal(scale=cfg.noise_sigma, size=cfg.n_test)
    y_train = X_train @ w_star + noise_train
    y_test = X_test @ w_star + noise_test
    return X_train, y_train, X_test, y_test, w_star


def _apply_corruption(cfg: SyntheticConfig, rng: np.random.Generator, X_train, y_train, w_star):
    n = X_train.shape[0]
    n_inliers = int(cfg.alpha * n)
    perm = rng.permutation(n)
    inlier_idx = perm[:n_inliers]
    outlier_idx = perm[n_inliers:]
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

    if cfg.corruption_model == "label_only":
        y_train[outlier_idx] = rng.normal(scale=10.0 * cfg.noise_sigma, size=len(outlier_idx))
    elif cfg.corruption_model == "feature_label":
        X_train[outlier_idx] = rng.standard_normal((len(outlier_idx), X_train.shape[1]))
        y_train[outlier_idx] = rng.normal(scale=10.0 * cfg.noise_sigma, size=len(outlier_idx))
    elif cfg.corruption_model == "leverage":
        fake_dir = rng.standard_normal(X_train.shape[1])
        fake_dir -= fake_dir.dot(w_star) * w_star
        fake_dir /= np.linalg.norm(fake_dir)
        X_train[outlier_idx] = cfg.leverage_tau * rng.standard_normal((len(outlier_idx), X_train.shape[1]))
        y_train[outlier_idx] = X_train[outlier_idx] @ fake_dir
    elif cfg.corruption_model == "adaptive":
        fake_dir = rng.standard_normal(X_train.shape[1])
        fake_dir -= fake_dir.dot(w_star) * w_star
        fake_dir /= np.linalg.norm(fake_dir)
        X_train[outlier_idx] = cfg.leverage_tau * rng.standard_normal((len(outlier_idx), X_train.shape[1]))
        y_train[outlier_idx] = X_train[outlier_idx] @ fake_dir + 5.0 * rng.normal(size=len(outlier_idx))
    elif cfg.corruption_model == "heavy_tailed":
        nu = cfg.heavy_tail_nu
        scales = np.sqrt(nu / rng.chisquare(nu, size=cfg.n_train))
        X_train[:] = X_train * scales[:, None]
        y_train[:] = X_train @ w_star + rng.normal(scale=cfg.noise_sigma, size=cfg.n_train)
    else:
        raise ValueError(f"Unknown corruption model: {cfg.corruption_model}")

    return X_train, y_train, inlier_mask


def _batched_bucket_assignment(n_samples: int, batch_size: int):
    n_batches = n_samples // batch_size
    bucket_indices = []
    bucket_signs = []
    for _ in range(1):
        buckets = []
        signs = []
        for b in range(n_batches):
            idx = np.arange(b * batch_size, (b + 1) * batch_size)
            buckets.append(idx)
            signs.append(np.ones_like(idx, dtype=float))
        bucket_indices.append(buckets)
        bucket_signs.append(signs)
    return bucket_indices, bucket_signs


def _load_real_dataset(cfg: RealDataConfig):
    if cfg.dataset_name == "E2006-tfidf":
        if cfg.libsvm_path is None:
            raise ValueError("libsvm_path is required for E2006-tfidf")
        X, y = load_svmlight_file(cfg.libsvm_path)
    else:
        dataset = fetch_openml(cfg.dataset_name, as_frame=False, parser="auto")
        X = dataset.data
        y = dataset.target
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if cfg.n_subsample is not None and cfg.n_subsample < X.shape[0]:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(X.shape[0], size=cfg.n_subsample, replace=False)
        X = X[idx]
        y = y[idx]
    return X, y


def _prepare_real_covariate(cfg: RealDataConfig, rng: np.random.Generator):
    X, y = _load_real_dataset(cfg)
    n, d = X.shape
    n_seed = min(5 * d, n)
    seed_idx = rng.choice(n, size=n_seed, replace=False)
    X_seed = X[seed_idx]
    y_seed = y[seed_idx]
    if sparse.issparse(X_seed):
        X_seed = X_seed.toarray()
    X_seed, _, mean, std = _standardize(X_seed, X_seed)
    X_dense = X.toarray() if sparse.issparse(X) else X
    X_std = (X_dense - mean) / std
    ridge = 1e-4
    w_star = np.linalg.solve(X_seed.T @ X_seed + ridge * np.eye(d), X_seed.T @ y_seed)
    y_hat = X_std @ w_star
    noise_scale = cfg.noise_sigma * np.std(y_hat)
    n_inliers = int(cfg.alpha * n)
    perm = rng.permutation(n)
    inlier_idx = perm[:n_inliers]
    outlier_idx = perm[n_inliers:]
    y_corrupt = np.array(y_hat)
    y_corrupt[inlier_idx] = y_hat[inlier_idx] + rng.normal(scale=noise_scale, size=len(inlier_idx))
    if cfg.corruption_model == "label_only":
        y_corrupt[outlier_idx] = rng.normal(scale=10.0 * noise_scale, size=len(outlier_idx))
    elif cfg.corruption_model == "feature_label":
        X_std[outlier_idx] = rng.standard_normal((len(outlier_idx), d))
        y_corrupt[outlier_idx] = rng.normal(scale=10.0 * noise_scale, size=len(outlier_idx))
    elif cfg.corruption_model == "foreign":
        if cfg.foreign_dataset is None:
            raise ValueError("foreign_dataset must be set for foreign corruption")
        X_foreign, _ = _load_real_dataset(
            RealDataConfig(dataset_name=cfg.foreign_dataset, n_subsample=len(outlier_idx), random_seed=cfg.random_seed)
        )
        X_foreign = X_foreign.toarray() if sparse.issparse(X_foreign) else X_foreign
        X_foreign = X_foreign[:, :d] if X_foreign.shape[1] >= d else np.pad(X_foreign, ((0, 0), (0, d - X_foreign.shape[1])))
        X_std[outlier_idx] = X_foreign[: len(outlier_idx)]
        y_corrupt[outlier_idx] = rng.normal(scale=10.0 * noise_scale, size=len(outlier_idx))
    else:
        raise ValueError(f"Unknown corruption model: {cfg.corruption_model}")

    X_train, X_test, y_train, y_test = X_std[: int(0.7 * n)], X_std[int(0.7 * n) :], y_corrupt[: int(0.7 * n)], y_hat[int(0.7 * n) :]
    inlier_mask = np.zeros(X_train.shape[0], dtype=bool)
    inlier_mask[inlier_idx[inlier_idx < X_train.shape[0]]] = True
    return X_train, y_train, X_test, y_test, w_star, inlier_mask


def _run_expander(cfg: BaseRunConfig, X_train, y_train, sketch_override: Optional[dict] = None):
    if isinstance(cfg, SyntheticConfig):
        params = cfg
    else:
        params = cfg
    ldr = ExpanderLDRRegressor(
        alpha=params.alpha,
        repetitions=params.repetitions,
        buckets=params.buckets,
        left_degree=params.left_degree,
        filtering_rounds=params.filtering_rounds,
        seeds=params.seeds,
        blocks=params.blocks,
        ridge=params.ridge,
        prune_eta=params.prune_eta,
        prune_rho=params.prune_rho,
        robust_method=params.robust_method,
        sketch_type=params.sketch_type,
        allow_replacement=params.allow_replacement,
        use_signs=params.use_signs,
        list_size_cap=params.list_size_cap,
        return_diagnostics=True,
        timing=True,
        random_state=params.random_seed,
    )
    if sketch_override:
        for key, value in sketch_override.items():
            setattr(ldr, key, value)
    ldr.fit(X_train, y_train)
    return ldr


def _countsketch_ols(X: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator):
    n, d = X.shape
    rows = rng.integers(0, B, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    S = np.zeros((B, n))
    S[rows, np.arange(n)] = signs
    Xs = S @ X
    ys = S @ y
    coef = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return coef


def _run_baselines(X_train, y_train, baselines: List[str], rng: np.random.Generator):
    results = {}
    for name in baselines:
        if name == "countsketch":
            coef = _countsketch_ols(X_train, y_train, B=min(200, X_train.shape[0]), rng=rng)
            results[name] = coef
            continue
        if name == "tukey":
            coef = _tukey_biweight(X_train, y_train)
            results[name] = coef
            continue
        if name == "lts":
            coef = _trimmed_least_squares(X_train, y_train)
            results[name] = coef
            continue
        if name == "sos_stub":
            results[name] = None
            continue
        model_cls = BASELINES.get(name)
        if model_cls is None:
            continue
        if name == "lad":
            model = model_cls(quantile=0.5, alpha=0.0)
        else:
            model = model_cls()
        model.fit(X_train, y_train)
        results[name] = model.coef_.astype(float)
    return results


def _tukey_biweight(X: np.ndarray, y: np.ndarray, max_iter: int = 20, c: float = 4.685):
    n, d = X.shape
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - X @ coef
        mad = np.median(np.abs(resid)) + 1e-8
        u = resid / (c * mad)
        w = (1 - u**2) ** 2
        w[np.abs(u) >= 1] = 0
        W = np.diag(w)
        coef = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
    return coef


def _trimmed_least_squares(X: np.ndarray, y: np.ndarray, trim_frac: float = 0.2, n_iter: int = 5):
    n = X.shape[0]
    keep = np.arange(n)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        resid = y - X @ coef
        keep = np.argsort(resid**2)[: int((1 - trim_frac) * n)]
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    return coef


def _evaluate_candidates(candidates: np.ndarray, X_test, y_test, w_star, X_train, y_train, alpha: float):
    param_errors = np.linalg.norm(candidates - w_star[None, :], axis=1)
    preds = candidates @ X_test.T
    risks = np.mean((preds - y_test[None, :]) ** 2, axis=1)
    oracle_idx = int(np.argmin(param_errors))
    oracle_error = float(param_errors[oracle_idx])
    oracle_risk = float(risks[oracle_idx])
    rank_curve = np.sort(param_errors)
    topk = {}
    success_threshold = np.sqrt(X_train.shape[1] / (alpha * X_train.shape[0]))
    for k in [1, 2, 5, 10]:
        kth_error = rank_curve[min(k - 1, len(rank_curve) - 1)]
        topk[k] = bool(kth_error <= success_threshold)
    sel_idx, _, sel_scores = select_by_median_squared_residual(X_train, y_train, candidates)
    sel_risk = float(risks[sel_idx])
    return {
        "oracle_param_error": oracle_error,
        "oracle_pred_risk": oracle_risk,
        "rank_curve": rank_curve.tolist(),
        "topk_success": topk,
        "selected_idx": int(sel_idx),
        "selected_pred_risk": sel_risk,
        "selection_scores": sel_scores.tolist(),
    }


def _save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _save_summary(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = sorted(rows[0].keys())
    summary_rows = []
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary_rows.append(
            {
                "metric": key,
                "median": float(np.median(arr)),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                "failure_rate": float(np.mean(~np.isfinite(arr))),
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "median", "iqr", "failure_rate"])
        writer.writeheader()
        writer.writerows(summary_rows)


def _config_to_dict(cfg: BaseRunConfig) -> Dict[str, Any]:
    if isinstance(cfg, SyntheticConfig):
        return cfg.__dict__.copy()
    if isinstance(cfg, RealDataConfig):
        return cfg.__dict__.copy()
    return cfg.__dict__.copy()


def _plot_curve(x, ys, labels, title, path):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel("x")
    plt.ylabel("metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_hist(values, title, path):
    plt.figure()
    plt.hist(values, bins=30, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_heatmap(matrix, title, path):
    plt.figure()
    plt.imshow(matrix, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_stacked_bar(labels, stacks, title, path):
    plt.figure()
    bottom = np.zeros(len(labels))
    for name, values in stacks.items():
        plt.bar(labels, values, bottom=bottom, label=name)
        bottom += np.array(values)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_scatter(points, labels, title, path):
    plt.figure()
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(points[mask, 0], points[mask, 1], label=str(label), alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _run_trial(exp_cfg: ExperimentConfig, trial_seed: int):
    rng = np.random.default_rng(trial_seed)
    if exp_cfg.data_type == "synthetic":
        cfg = exp_cfg.config
        assert isinstance(cfg, SyntheticConfig)
        X_train, y_train, X_test, y_test, w_star = _generate_gaussian_data(cfg, rng)
        X_train, y_train, inlier_mask = _apply_corruption(cfg, rng, X_train, y_train, w_star)
    else:
        cfg = exp_cfg.config
        assert isinstance(cfg, RealDataConfig)
        X_train, y_train, X_test, y_test, w_star, inlier_mask = _prepare_real_covariate(cfg, rng)

    X_train, X_test, mean, std = _standardize(X_train, X_test)

    sketch_override = None
    if exp_cfg.extras.get("fixed_batches"):
        batch_size = exp_cfg.extras.get("batch_size", 32)
        bucket_indices, bucket_signs = _batched_bucket_assignment(X_train.shape[0], batch_size)
        sketch_override = {
            "sketch_type": "fixed",
            "fixed_bucket_indices": bucket_indices,
            "fixed_bucket_signs": bucket_signs,
        }

    ldr = _run_expander(cfg, X_train, y_train, sketch_override=sketch_override)
    candidates = ldr.candidates_
    metrics = _evaluate_candidates(
        candidates, X_test, y_test, w_star, X_train, y_train, cfg.alpha
    )
    baselines = _run_baselines(X_train, y_train, exp_cfg.baselines, rng)

    baseline_metrics = {}
    for name, coef in baselines.items():
        if coef is None:
            baseline_metrics[f"baseline_{name}"] = "not_implemented"
            continue
        pred = X_test @ coef
        baseline_metrics[f"baseline_{name}_risk"] = float(np.mean((pred - y_test) ** 2))
        baseline_metrics[f"baseline_{name}_param_error"] = float(np.linalg.norm(coef - w_star))

    diagnostic_summary = {}
    contamination_stats = None
    pruning_stats = None
    if hasattr(ldr, "diagnostics_") and ldr.diagnostics_:
        filtering_info = ldr.diagnostics_[0]["filtering_info"]
        diagnostic_summary = {
            "rounds_ran": filtering_info.get("rounds_ran"),
            "n_active": filtering_info.get("n_active"),
            "lambda_max": filtering_info.get("lambda_max"),
            "target_var": filtering_info.get("target_var"),
            "pruned_counts": [
                len(pruned) for pruned in filtering_info.get("pruned_buckets_by_round", [])
            ],
            "bucket_score_histograms": [
                (hist[0].tolist(), hist[1].tolist())
                for hist in filtering_info.get("bucket_score_histograms", [])
            ],
        }
        if exp_cfg.name in {"E5A", "E5B", "E5C"}:
            bucket_indices = ldr.diagnostics_[0]["bucket_indices"]
            contamination_stats_raw = bucket_contamination_stats(bucket_indices, inlier_mask)
            outlier_fraction = {
                tuple(bid): float(frac)
                for bid, frac in zip(
                    contamination_stats_raw["bucket_ids"],
                    contamination_stats_raw["outlier_fraction"],
                )
            }
            pruned = filtering_info.get("pruned_buckets_by_round", [])
            pruning_stats_raw = pruning_precision_recall(pruned, outlier_fraction)
            contamination_stats = {
                "per_bucket_inlier_count": contamination_stats_raw[
                    "per_bucket_inlier_count"
                ].tolist(),
                "per_bucket_outlier_count": contamination_stats_raw[
                    "per_bucket_outlier_count"
                ].tolist(),
                "outlier_fraction": contamination_stats_raw["outlier_fraction"].tolist(),
                "fraction_lightly_contaminated": contamination_stats_raw[
                    "fraction_lightly_contaminated"
                ],
                "unique_neighbor_stats_for_inliers": contamination_stats_raw[
                    "unique_neighbor_stats_for_inliers"
                ],
            }
            pruning_stats = {
                "precision": pruning_stats_raw["precision"].tolist(),
                "recall": pruning_stats_raw["recall"].tolist(),
            }

    pca_points = None
    pca_labels = None
    if exp_cfg.name == "E6B" and candidates.shape[0] >= 2:
        pca = PCA(n_components=2)
        pca_points = pca.fit_transform(candidates).tolist()
        pca_labels = list(range(candidates.shape[0]))

    return {
        "experiment": exp_cfg.name,
        "variant": exp_cfg.variant,
        "trial_seed": trial_seed,
        "oracle_param_error": metrics["oracle_param_error"],
        "oracle_pred_risk": metrics["oracle_pred_risk"],
        "selected_pred_risk": metrics["selected_pred_risk"],
        "list_size": int(candidates.shape[0]),
        "timing_fit_total": ldr.timings_.get("fit_total") if hasattr(ldr, "timings_") else None,
        "timing_sketch": ldr.timings_.get("sketch") if hasattr(ldr, "timings_") else None,
        "timing_filtering": ldr.timings_.get("filtering") if hasattr(ldr, "timings_") else None,
        "timing_aggregation": ldr.timings_.get("aggregation") if hasattr(ldr, "timings_") else None,
        "timing_solve": ldr.timings_.get("solve") if hasattr(ldr, "timings_") else None,
        "timing_clustering": ldr.timings_.get("clustering") if hasattr(ldr, "timings_") else None,
        "nnz": int(np.count_nonzero(X_train)),
        "rank_curve": metrics["rank_curve"],
        "topk_success": metrics["topk_success"],
        "selection_scores": metrics["selection_scores"],
        "filtering_diagnostics": diagnostic_summary,
        "contamination_stats": contamination_stats,
        "pruning_stats": pruning_stats,
        "pca_points": pca_points,
        "pca_labels": pca_labels,
        **baseline_metrics,
    }


def run_experiment(name: str, outdir: str, n_trials: Optional[int] = None, seed: int = 0):
    configs = experiment_registry().get(name)
    if configs is None:
        raise ValueError(f"Unknown experiment {name}")
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for exp_cfg in configs:
        cfg = exp_cfg.config
        trials = n_trials if n_trials is not None else cfg.n_trials
        config_dump = {
            "experiment": exp_cfg.name,
            "variant": exp_cfg.variant,
            "config": _config_to_dict(cfg),
            "baselines": exp_cfg.baselines,
            "extras": exp_cfg.extras,
        }
        with (outdir_path / f"{exp_cfg.name}_{exp_cfg.variant}_config.json").open("w") as f:
            json.dump(config_dump, f, indent=2)
        for trial in range(trials):
            trial_seed = seed + trial
            row = _run_trial(exp_cfg, trial_seed)
            all_rows.append(row)
    _save_jsonl(outdir_path / "results.jsonl", all_rows)
    numeric_rows = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))} for row in all_rows
    ]
    _save_summary(outdir_path / "summary.csv", numeric_rows)

    if all_rows:
        errors = [row["oracle_param_error"] for row in all_rows]
        _plot_hist(errors, f"{name} oracle param error", plots_dir / f"{name}_oracle_error_hist.png")
        if name in {"E1A", "E1B"}:
            nnz_vals = [row["nnz"] for row in all_rows]
            plt.figure()
            plt.loglog(nnz_vals, errors, marker="o")
            plt.xlabel("nnz(X)")
            plt.ylabel("oracle_param_error")
            plt.title(f"{name} scaling")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{name}_loglog.png")
            plt.close()
        if name == "E1C":
            mean_success = np.mean([row["topk_success"][1] for row in all_rows])
            _plot_heatmap(
                np.array([[mean_success]]),
                "E1C success heatmap",
                plots_dir / "E1C_heatmap.png",
            )
        if name == "E2B":
            diag = all_rows[0].get("filtering_diagnostics", {})
            lambda_max = diag.get("lambda_max", [])
            target_var = diag.get("target_var", [])
            if lambda_max:
                plt.figure()
                plt.plot(lambda_max, label="lambda_max")
                plt.plot(target_var, label="target_var")
                plt.xlabel("round")
                plt.ylabel("value")
                plt.title("Filtering waterfall")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "E2B_filtering_waterfall.png")
                plt.close()
            histograms = diag.get("bucket_score_histograms", [])
            if histograms:
                counts, edges = histograms[-1]
                plt.figure()
                centers = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
                plt.bar(centers, counts, width=centers[1] - centers[0])
                plt.title("Bucket score histogram")
                plt.tight_layout()
                plt.savefig(plots_dir / "E2B_bucket_scores.png")
                plt.close()
        if name == "E6B":
            pca_points = all_rows[0].get("pca_points")
            pca_labels = all_rows[0].get("pca_labels")
            if pca_points is not None and pca_labels is not None:
                points = np.array(pca_points)
                labels = np.array(pca_labels)
                _plot_scatter(
                    points,
                    labels,
                    "E6B PCA candidate scatter",
                    plots_dir / "E6B_pca_scatter.png",
                )
        if name == "E7A":
            nnz_vals = [row["nnz"] for row in all_rows]
            sketch_times = [row["timing_sketch"] for row in all_rows]
            solve_times = [row["timing_solve"] for row in all_rows]
            _plot_curve(
                nnz_vals,
                [sketch_times, solve_times],
                ["sketch", "solve"],
                "E7A runtime scaling",
                plots_dir / "E7A_runtime.png",
            )
        if name == "E7B":
            timing_keys = ["timing_sketch", "timing_aggregation", "timing_solve"]
            averages = {
                key.replace("timing_", ""): float(
                    np.mean([row[key] for row in all_rows if row[key] is not None])
                )
                for key in timing_keys
            }
            _plot_stacked_bar(
                ["total"],
                {k: [v] for k, v in averages.items()},
                "E7B time breakdown",
                plots_dir / "E7B_breakdown.png",
            )
    return all_rows


def run_config(exp_cfg: ExperimentConfig, outdir: str, n_trials: Optional[int] = None, seed: int = 0):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cfg = exp_cfg.config
    trials = n_trials if n_trials is not None else cfg.n_trials
    with (outdir_path / f"{exp_cfg.name}_{exp_cfg.variant}_config.json").open("w") as f:
        json.dump(
            {
                "experiment": exp_cfg.name,
                "variant": exp_cfg.variant,
                "config": _config_to_dict(cfg),
                "baselines": exp_cfg.baselines,
                "extras": exp_cfg.extras,
            },
            f,
            indent=2,
        )
    rows = []
    for trial in range(trials):
        trial_seed = seed + trial
        rows.append(_run_trial(exp_cfg, trial_seed))
    _save_jsonl(outdir_path / "results.jsonl", rows)
    numeric_rows = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))} for row in rows
    ]
    _save_summary(outdir_path / "summary.csv", numeric_rows)
    if rows:
        errors = [row["oracle_param_error"] for row in rows]
        _plot_hist(errors, f"{exp_cfg.name} oracle param error", plots_dir / f"{exp_cfg.name}_oracle_error_hist.png")
    return rows


def experiment_registry() -> Dict[str, List[ExperimentConfig]]:
    return {
        "E1A": [
            ExperimentConfig(
                name="E1A",
                variant="scaling",
                data_type="synthetic",
                config=SyntheticConfig(n_train=2000, n_test=1000, d=20, alpha=0.2),
            )
        ],
        "E1B": [
            ExperimentConfig(
                name="E1B",
                variant="phase_transition",
                data_type="synthetic",
                config=SyntheticConfig(n_train=4000, n_test=1000, d=40, alpha=0.2),
            )
        ],
        "E1C": [
            ExperimentConfig(
                name="E1C",
                variant="heatmap",
                data_type="synthetic",
                config=SyntheticConfig(n_train=3000, n_test=1000, d=30, alpha=0.2),
            )
        ],
        "E2A": [
            ExperimentConfig(
                name="E2A",
                variant="label_only",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="label_only"),
            )
        ],
        "E2B": [
            ExperimentConfig(
                name="E2B",
                variant="leverage",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="leverage", leverage_tau=8.0),
            )
        ],
        "E2C": [
            ExperimentConfig(
                name="E2C",
                variant="adaptive",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="adaptive"),
                baselines=["countsketch"],
            )
        ],
        "E2D": [
            ExperimentConfig(
                name="E2D",
                variant="heavy_tailed",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="heavy_tailed", heavy_tail_nu=3),
            )
        ],
        "E3A": [
            ExperimentConfig(
                name="E3A",
                variant="sos_stub",
                data_type="synthetic",
                config=SyntheticConfig(d=10),
                baselines=["sos_stub"],
            )
        ],
        "E3B": [
            ExperimentConfig(
                name="E3B",
                variant="true_batch",
                data_type="synthetic",
                config=SyntheticConfig(),
                extras={"fixed_batches": True, "batch_size": 32},
            )
        ],
        "E3C": [
            ExperimentConfig(
                name="E3C",
                variant="baselines",
                data_type="synthetic",
                config=SyntheticConfig(),
                baselines=["huber", "theilsen", "ransac", "lad", "tukey", "lts"],
            )
        ],
        "E4A": [
            ExperimentConfig(
                name="E4A",
                variant="sketch_ablation",
                data_type="synthetic",
                config=SyntheticConfig(sketch_type="expander"),
            ),
            ExperimentConfig(
                name="E4A",
                variant="one_hash",
                data_type="synthetic",
                config=SyntheticConfig(sketch_type="one_hash"),
            ),
            ExperimentConfig(
                name="E4A",
                variant="countsketch",
                data_type="synthetic",
                config=SyntheticConfig(sketch_type="countsketch"),
            ),
        ],
        "E4B": [
            ExperimentConfig(
                name="E4B",
                variant="mean",
                data_type="synthetic",
                config=SyntheticConfig(robust_method="mean"),
            ),
            ExperimentConfig(
                name="E4B",
                variant="mom",
                data_type="synthetic",
                config=SyntheticConfig(robust_method="mom"),
            ),
        ],
        "E4C": [
            ExperimentConfig(
                name="E4C",
                variant="filtering_on",
                data_type="synthetic",
                config=SyntheticConfig(filtering_rounds=4),
            ),
            ExperimentConfig(
                name="E4C",
                variant="filtering_off",
                data_type="synthetic",
                config=SyntheticConfig(filtering_rounds=0),
            ),
        ],
        "E4D": [
            ExperimentConfig(
                name="E4D",
                variant="signs_on",
                data_type="synthetic",
                config=SyntheticConfig(use_signs=True),
            ),
            ExperimentConfig(
                name="E4D",
                variant="signs_off",
                data_type="synthetic",
                config=SyntheticConfig(use_signs=False),
            ),
        ],
        "E4E": [
            ExperimentConfig(
                name="E4E",
                variant="seed_sweep",
                data_type="synthetic",
                config=SyntheticConfig(seeds=4),
            )
        ],
        "E5A": [
            ExperimentConfig(
                name="E5A",
                variant="contamination",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="leverage"),
            )
        ],
        "E5B": [
            ExperimentConfig(
                name="E5B",
                variant="pruning_precision",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="leverage"),
            )
        ],
        "E5C": [
            ExperimentConfig(
                name="E5C",
                variant="potential",
                data_type="synthetic",
                config=SyntheticConfig(corruption_model="leverage"),
            )
        ],
        "E6A": [
            ExperimentConfig(
                name="E6A",
                variant="real_covariate",
                data_type="real",
                config=RealDataConfig(dataset_name="YearPredictionMSD", corruption_model="label_only"),
            )
        ],
        "E6B": [
            ExperimentConfig(
                name="E6B",
                variant="mixture",
                data_type="real",
                config=RealDataConfig(dataset_name="Online News Popularity", corruption_model="feature_label"),
            )
        ],
        "E7A": [
            ExperimentConfig(
                name="E7A",
                variant="runtime_scaling",
                data_type="synthetic",
                config=SyntheticConfig(n_train=5000, n_test=1000, d=50),
            )
        ],
        "E7B": [
            ExperimentConfig(
                name="E7B",
                variant="breakdown",
                data_type="synthetic",
                config=SyntheticConfig(n_train=3000, n_test=1000, d=30),
            )
        ],
        "E8A": [
            ExperimentConfig(
                name="E8A",
                variant="alpha_misspec",
                data_type="synthetic",
                config=SyntheticConfig(alpha=0.15),
            )
        ],
        "E8B": [
            ExperimentConfig(
                name="E8B",
                variant="hyperparam",
                data_type="synthetic",
                config=SyntheticConfig(buckets=150, repetitions=6),
            )
        ],
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Run expander LDR experiments")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_experiment(args.experiment, args.outdir, n_trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
