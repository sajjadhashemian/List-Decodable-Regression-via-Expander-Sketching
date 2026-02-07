from __future__ import annotations

import argparse
import csv
import inspect
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
    LinearRegression,
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
    n_trials: int = 30
    outdir: str = "results"
    save_raw: bool = True


@dataclass
class SyntheticConfig(BaseRunConfig):
    n_train: int = 5000
    n_test: int = 2000
    d: int = 20
    alpha: float = 0.2
    alpha_hat: Optional[float] = None
    noise_sigma: float = 0.1
    corruption_model: str = "label_only"
    label_corruption: str = "gaussian"
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
    enable_filtering: bool = True
    success_threshold_mult: float = 1.0
    sparse_nnz_per_row: Optional[int] = None


@dataclass
class RealDataConfig(BaseRunConfig):
    dataset_name: str = "YearPredictionMSD"
    n_subsample: Optional[int] = None
    alpha: float = 0.2
    alpha_hat: Optional[float] = None
    corruption_model: str = "label_only"
    noise_sigma: float = 0.05
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
    enable_filtering: bool = True
    foreign_dataset: Optional[str] = None
    libsvm_path: Optional[str] = None
    mixture_alpha: Optional[float] = None


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


def _standardize(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std, mean, std


def _add_intercept(X: np.ndarray) -> np.ndarray:
    intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, intercept])


def _generate_gaussian_data(cfg: SyntheticConfig, rng: np.random.Generator):
    if cfg.sparse_nnz_per_row is None:
        X_train = rng.standard_normal((cfg.n_train, cfg.d))
        X_test = rng.standard_normal((cfg.n_test, cfg.d))
    else:
        d = cfg.d
        X_train = np.zeros((cfg.n_train, d))
        X_test = np.zeros((cfg.n_test, d))
        for X in [X_train, X_test]:
            rows = np.arange(X.shape[0])
            cols = rng.integers(0, d, size=(X.shape[0], cfg.sparse_nnz_per_row))
            values = rng.standard_normal(size=(X.shape[0], cfg.sparse_nnz_per_row))
            X[rows[:, None], cols] = values
    w_star = rng.standard_normal(cfg.d)
    w_star = w_star / np.linalg.norm(w_star)
    noise_train = rng.normal(scale=cfg.noise_sigma, size=cfg.n_train)
    noise_test = rng.normal(scale=cfg.noise_sigma, size=cfg.n_test)
    y_train = X_train @ w_star + noise_train
    y_test = X_test @ w_star + noise_test
    return X_train, y_train, X_test, y_test, w_star


def _label_only_corruption(
    mode: str,
    rng: np.random.Generator,
    X: np.ndarray,
    w_star: np.ndarray,
    y_outliers: np.ndarray,
):
    if mode == "uniform":
        y_outliers[:] = rng.uniform(-10, 10, size=y_outliers.shape[0])
    elif mode == "gaussian":
        y_outliers[:] = rng.normal(scale=5.0, size=y_outliers.shape[0])
    elif mode == "sign_flip":
        y_outliers[:] = -y_outliers
    elif mode == "fake_model":
        fake_dir = rng.standard_normal(X.shape[1])
        fake_dir -= fake_dir.dot(w_star) * w_star
        fake_dir /= np.linalg.norm(fake_dir)
        y_outliers[:] = X @ fake_dir
    else:
        raise ValueError(f"Unknown label corruption mode: {mode}")


def _apply_corruption(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_star: np.ndarray,
):
    n = X_train.shape[0]
    n_inliers = int(cfg.alpha * n)
    perm = rng.permutation(n)
    inlier_idx = perm[:n_inliers]
    outlier_idx = perm[n_inliers:]
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

    if cfg.corruption_model == "label_only":
        y_outliers = y_train[outlier_idx]
        _label_only_corruption(cfg.label_corruption, rng, X_train[outlier_idx], w_star, y_outliers)
        y_train[outlier_idx] = y_outliers
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
        y_train[outlier_idx] = rng.normal(scale=10.0 * cfg.noise_sigma, size=len(outlier_idx))
    else:
        raise ValueError(f"Unknown corruption model: {cfg.corruption_model}")

    return X_train, y_train, inlier_mask


def _generate_batched_data(cfg: SyntheticConfig, rng: np.random.Generator, batch_size: int):
    n_batches = cfg.n_train // batch_size
    n = n_batches * batch_size
    X_train = rng.standard_normal((n, cfg.d))
    w_star = rng.standard_normal(cfg.d)
    w_star = w_star / np.linalg.norm(w_star)
    y_train = X_train @ w_star + rng.normal(scale=cfg.noise_sigma, size=n)
    inlier_mask = np.zeros(n, dtype=bool)

    n_clean_batches = int(cfg.alpha * n_batches)
    batch_perm = rng.permutation(n_batches)
    clean_batches = set(batch_perm[:n_clean_batches].tolist())
    fake_dir = rng.standard_normal(cfg.d)
    fake_dir -= fake_dir.dot(w_star) * w_star
    fake_dir /= np.linalg.norm(fake_dir)

    for b in range(n_batches):
        idx = slice(b * batch_size, (b + 1) * batch_size)
        if b in clean_batches:
            inlier_mask[idx] = True
        else:
            X_train[idx] = cfg.leverage_tau * rng.standard_normal((batch_size, cfg.d))
            y_train[idx] = X_train[idx] @ fake_dir
    X_test = rng.standard_normal((cfg.n_test, cfg.d))
    y_test = X_test @ w_star + rng.normal(scale=cfg.noise_sigma, size=cfg.n_test)
    return X_train, y_train, X_test, y_test, w_star, inlier_mask


def _batched_bucket_assignment(n_samples: int, batch_size: int, repetitions: int):
    n_batches = int(np.ceil(n_samples / batch_size))
    bucket_indices = []
    bucket_signs = []
    for _ in range(repetitions):
        buckets = []
        signs = []
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_samples)
            idx = np.arange(start, end)
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
    n_seed = min(5 * d, 2000, n)
    seed_idx = rng.choice(n, size=n_seed, replace=False)
    X_seed = X[seed_idx]
    y_seed = y[seed_idx]
    X_seed = X_seed.toarray() if sparse.issparse(X_seed) else np.asarray(X_seed)
    mean = X_seed.mean(axis=0)
    std = X_seed.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    X_seed_std = (X_seed - mean) / std
    X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X_std = (X_dense - mean) / std
    ridge = 1e-4
    w_star = np.linalg.solve(X_seed_std.T @ X_seed_std + ridge * np.eye(d), X_seed_std.T @ y_seed)
    y_hat = X_std @ w_star
    noise_scale = cfg.noise_sigma * np.std(y_hat)

    n_inliers = int(cfg.alpha * n)
    perm = rng.permutation(n)
    inlier_idx = perm[:n_inliers]
    outlier_idx = perm[n_inliers:]
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

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
        foreign_cfg = RealDataConfig(dataset_name=cfg.foreign_dataset, n_subsample=len(outlier_idx), random_seed=cfg.random_seed)
        X_foreign, _ = _load_real_dataset(foreign_cfg)
        X_foreign = X_foreign.toarray() if sparse.issparse(X_foreign) else X_foreign
        if X_foreign.shape[1] < d:
            X_foreign = np.pad(X_foreign, ((0, 0), (0, d - X_foreign.shape[1])))
        X_std[outlier_idx] = X_foreign[: len(outlier_idx), :d]
        y_corrupt[outlier_idx] = rng.normal(scale=10.0 * noise_scale, size=len(outlier_idx))
    else:
        raise ValueError(f"Unknown corruption model: {cfg.corruption_model}")

    split = int(0.7 * n)
    X_train = X_std[:split]
    X_test = X_std[split:]
    y_train = y_corrupt[:split]
    y_test = y_hat[split:]
    inlier_mask_train = inlier_mask[:split]
    return X_train, y_train, X_test, y_test, w_star, inlier_mask_train


def _prepare_real_mixture(cfg: RealDataConfig, rng: np.random.Generator):
    if cfg.mixture_alpha is None:
        raise ValueError("mixture_alpha must be set for mixture experiments")
    X, y = _load_real_dataset(cfg)
    n, d = X.shape
    X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    n_seed = min(5 * d, 2000, n)
    seed_idx = rng.choice(n, size=2 * n_seed, replace=False)
    seed_1 = seed_idx[:n_seed]
    seed_2 = seed_idx[n_seed:]
    mean = X_dense.mean(axis=0)
    std = X_dense.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    X_std = (X_dense - mean) / std

    ridge = 1e-4
    w1 = np.linalg.solve(
        X_std[seed_1].T @ X_std[seed_1] + ridge * np.eye(d),
        X_std[seed_1].T @ y[seed_1],
    )
    w2 = np.linalg.solve(
        X_std[seed_2].T @ X_std[seed_2] + ridge * np.eye(d),
        X_std[seed_2].T @ y[seed_2],
    )
    w1 = w1 / np.linalg.norm(w1)
    w2 = w2 / np.linalg.norm(w2)

    n1 = int(cfg.alpha * n)
    n2 = int(cfg.mixture_alpha * n)
    perm = rng.permutation(n)
    idx1 = perm[:n1]
    idx2 = perm[n1 : n1 + n2]
    outlier_idx = perm[n1 + n2 :]
    y = np.zeros(n)
    y[idx1] = X_std[idx1] @ w1
    y[idx2] = X_std[idx2] @ w2
    y[outlier_idx] = rng.normal(scale=2.0, size=len(outlier_idx))
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[idx1] = True

    split = int(0.7 * n)
    return X_std[:split], y[:split], X_std[split:], X_std[split:] @ w1, w1, inlier_mask[:split], w2


def _run_expander(cfg: BaseRunConfig, X_train, y_train, sketch_override: Optional[dict] = None):
    params = cfg
    alpha_hat = params.alpha_hat if params.alpha_hat is not None else params.alpha
    ldr = ExpanderLDRRegressor(
        alpha=alpha_hat,
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
        enable_filtering=params.enable_filtering,
        random_state=params.random_seed,
    )
    if sketch_override:
        for key, value in sketch_override.items():
            setattr(ldr, key, value)
    ldr.fit(X_train, y_train)
    return ldr


def _countsketch_ols(X: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator):
    n, _ = X.shape
    rows = rng.integers(0, B, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    S = np.zeros((B, n))
    S[rows, np.arange(n)] = signs
    Xs = S @ X
    ys = S @ y
    coef = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return coef


def _jl_ols(X: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator):
    """Dense JL baseline kept outside the bucket-moment decomposition."""
    G = rng.normal(scale=1.0 / np.sqrt(B), size=(B, X.shape[0]))
    Xs = G @ X
    ys = G @ y
    coef = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return coef


def _batched_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    ridge: float,
) -> np.ndarray:
    n = X_train.shape[0]
    n_batches = n // batch_size
    coefs = []
    for b in range(n_batches):
        idx = slice(b * batch_size, (b + 1) * batch_size)
        Xb = X_train[idx]
        yb = y_train[idx]
        H = Xb.T @ Xb + ridge * np.eye(X_train.shape[1])
        g = Xb.T @ yb
        coefs.append(np.linalg.solve(H, g))
    return np.median(np.stack(coefs, axis=0), axis=0)


def _run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baselines: List[str],
    rng: np.random.Generator,
    extras: Optional[Dict[str, Any]] = None,
):
    results = {}
    for name in baselines:
        if name == "countsketch":
            coef = _countsketch_ols(X_train, y_train, B=min(200, X_train.shape[0]), rng=rng)
            results[name] = coef
            continue
        if name == "jl_ols":
            coef = _jl_ols(X_train, y_train, B=min(200, X_train.shape[0]), rng=rng)
            results[name] = coef
            continue
        if name == "batched":
            if extras is None or "batch_size" not in extras:
                raise ValueError("batched baseline requires batch_size in extras")
            coef = _batched_baseline(X_train, y_train, extras["batch_size"], ridge=1e-3)
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
        X_base = _add_intercept(X_train)
        if name == "lad":
            model = model_cls(quantile=0.5, alpha=0.0, fit_intercept=False)
        elif name == "ransac":
            base = LinearRegression(fit_intercept=False)
            signature = inspect.signature(model_cls)
            if "estimator" in signature.parameters:
                model = model_cls(estimator=base)
            else:
                model = model_cls(base_estimator=base)
        else:
            model = model_cls(fit_intercept=False)
        model.fit(X_base, y_train)
        coef = model.coef_.astype(float)
        results[name] = coef
    return results


def _tukey_biweight(X: np.ndarray, y: np.ndarray, max_iter: int = 20, c: float = 4.685):
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
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        resid = y - X @ coef
        keep = np.argsort(resid**2)[: int((1 - trim_frac) * n)]
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    return coef


def _evaluate_candidates(
    candidates: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_star: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    threshold_mult: float,
):
    param_errors = np.linalg.norm(candidates - w_star[None, :], axis=1)
    preds = candidates @ X_test.T
    risks = np.mean((preds - y_test[None, :]) ** 2, axis=1)
    oracle_idx = int(np.argmin(param_errors))
    oracle_error = float(param_errors[oracle_idx])
    oracle_risk = float(risks[oracle_idx])
    rank_curve = np.sort(param_errors)
    topk = {}
    success_threshold = threshold_mult * np.sqrt(X_train.shape[1] / (alpha * X_train.shape[0]))
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
    return cfg.__dict__.copy()


def _plot_curve(x, ys, labels, title, path, xscale: str = "linear", yscale: str = "linear"):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("x")
    plt.ylabel("metric")
    plt.xscale(xscale)
    plt.yscale(yscale)
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


def _plot_heatmap(matrix, title, path, x_labels=None, y_labels=None):
    plt.figure()
    plt.imshow(matrix, origin="lower", aspect="auto")
    if x_labels is not None:
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    if y_labels is not None:
        plt.yticks(np.arange(len(y_labels)), y_labels)
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


def _plot_with_iqr(x, medians, iqrs, title, path, xscale="log", yscale="log"):
    plt.figure()
    medians = np.array(medians)
    iqrs = np.array(iqrs)
    plt.plot(x, medians, marker="o")
    plt.fill_between(x, medians - 0.5 * iqrs, medians + 0.5 * iqrs, alpha=0.2)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel("x")
    plt.ylabel("oracle_param_error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _median_rank_curve(rows: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    curves = [row["rank_curve"] for row in rows if row.get("rank_curve")]
    if not curves:
        return None
    min_len = min(len(curve) for curve in curves)
    stacked = np.stack([np.array(curve[:min_len]) for curve in curves], axis=0)
    return np.median(stacked, axis=0)


def _run_trial(exp_cfg: ExperimentConfig, trial_seed: int):
    rng = np.random.default_rng(trial_seed)
    if exp_cfg.data_type == "synthetic":
        cfg = exp_cfg.config
        assert isinstance(cfg, SyntheticConfig)
        if exp_cfg.extras.get("batched"):
            batch_size = exp_cfg.extras.get("batch_size", 32)
            X_train, y_train, X_test, y_test, w_star, inlier_mask = _generate_batched_data(cfg, rng, batch_size)
        else:
            X_train, y_train, X_test, y_test, w_star = _generate_gaussian_data(cfg, rng)
            X_train, y_train, inlier_mask = _apply_corruption(cfg, rng, X_train, y_train, w_star)
        threshold_mult = cfg.success_threshold_mult
    else:
        cfg = exp_cfg.config
        assert isinstance(cfg, RealDataConfig)
        if exp_cfg.extras.get("mixture"):
            X_train, y_train, X_test, y_test, w_star, inlier_mask, w_star_alt = _prepare_real_mixture(cfg, rng)
        else:
            X_train, y_train, X_test, y_test, w_star, inlier_mask = _prepare_real_covariate(cfg, rng)
        threshold_mult = 1.0

    X_train, X_test, _, _ = _standardize(X_train, X_test)

    sketch_override = None
    if exp_cfg.extras.get("fixed_batches"):
        batch_size = exp_cfg.extras.get("batch_size", 32)
        bucket_indices, bucket_signs = _batched_bucket_assignment(
            X_train.shape[0], batch_size, cfg.repetitions
        )
        sketch_override = {
            "sketch_type": "fixed",
            "fixed_bucket_indices": bucket_indices,
            "fixed_bucket_signs": bucket_signs,
        }

    ldr = _run_expander(cfg, X_train, y_train, sketch_override=sketch_override)
    candidates = ldr.candidates_
    metrics = _evaluate_candidates(
        candidates, X_test, y_test, w_star, X_train, y_train, cfg.alpha, threshold_mult
    )
    baselines = _run_baselines(X_train, y_train, exp_cfg.baselines, rng, exp_cfg.extras)

    baseline_metrics = {}
    for name, coef in baselines.items():
        if coef is None:
            baseline_metrics[f"baseline_{name}"] = "not_implemented"
            continue
        pred = _add_intercept(X_test) @ coef if name in BASELINES else X_test @ coef
        baseline_metrics[f"baseline_{name}_risk"] = float(np.mean((pred - y_test) ** 2))
        coef_vec = coef[:-1] if coef.shape[0] == w_star.shape[0] + 1 else coef
        baseline_metrics[f"baseline_{name}_param_error"] = float(
            np.linalg.norm(coef_vec - w_star)
        )

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
        if exp_cfg.name in {"E4A", "E5A", "E5B", "E5C"}:
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
        "alpha": cfg.alpha,
        "d": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_over_d": float(round(X_train.shape[0] / X_train.shape[1], 6)),
        "buckets": getattr(cfg, "buckets", None),
        "repetitions": getattr(cfg, "repetitions", None),
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
        _plot_hist(
            [row["oracle_param_error"] for row in all_rows],
            f"{name} oracle param error",
            plots_dir / f"{name}_oracle_error_hist.png",
        )
        _plot_experiment_figures(name, all_rows, plots_dir)
    return all_rows


def run_config(
    exp_cfg: ExperimentConfig, outdir: str, n_trials: Optional[int] = None, seed: int = 0
):
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
        _plot_hist(
            [row["oracle_param_error"] for row in rows],
            f"{exp_cfg.name} oracle param error",
            plots_dir / f"{exp_cfg.name}_oracle_error_hist.png",
        )
    return rows


def _aggregate_by(rows: List[Dict[str, Any]], key: str, metric: str):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], {}).setdefault(row[key], []).append(row[metric])
    return grouped


def _plot_experiment_figures(name: str, rows: List[Dict[str, Any]], plots_dir: Path):
    if name == "E1A":
        grouped = _aggregate_by(rows, "n_train", "oracle_param_error")
        for variant, data in grouped.items():
            x = sorted(data.keys())
            medians = [np.median(data[val]) for val in x]
            iqrs = [np.percentile(data[val], 75) - np.percentile(data[val], 25) for val in x]
            _plot_with_iqr(
                x,
                medians,
                iqrs,
                f"{name} {variant} scaling",
                plots_dir / f"{name}_{variant}_loglog.png",
            )
    if name == "E1B":
        grouped = _aggregate_by(rows, "d", "oracle_param_error")
        for variant, data in grouped.items():
            x = sorted(data.keys())
            medians = [np.median(data[val]) for val in x]
            iqrs = [np.percentile(data[val], 75) - np.percentile(data[val], 25) for val in x]
            _plot_with_iqr(
                x,
                medians,
                iqrs,
                f"{name} {variant} scaling",
                plots_dir / f"{name}_{variant}_loglog.png",
            )
    if name == "E1C":
        alphas = sorted({row["alpha"] for row in rows})
        ratios = sorted({row["n_over_d"] for row in rows})
        matrix = np.zeros((len(alphas), len(ratios)))
        error_matrix = np.zeros((len(alphas), len(ratios)))
        for i, alpha in enumerate(alphas):
            for j, ratio in enumerate(ratios):
                subset = [
                    row for row in rows if row["alpha"] == alpha and row["n_over_d"] == ratio
                ]
                if subset:
                    matrix[i, j] = np.mean([row["topk_success"][1] for row in subset])
                    error_matrix[i, j] = np.median([row["oracle_param_error"] for row in subset])
        _plot_heatmap(matrix, "E1C success probability", plots_dir / "E1C_heatmap.png", ratios, alphas)
        _plot_heatmap(error_matrix, "E1C median error", plots_dir / "E1C_error_heatmap.png", ratios, alphas)
    if name == "E2B":
        alphas = sorted({row["alpha"] for row in rows})
        taus = sorted({row["variant"].split("tau_")[-1] for row in rows})
        for alpha in alphas:
            errors = []
            for tau in taus:
                subset = [
                    row for row in rows if row["variant"] == f"alpha_{alpha}_tau_{tau}"
                ]
                if subset:
                    errors.append(
                        np.median([row["oracle_param_error"] for row in subset])
                    )
            _plot_curve(
                taus,
                [errors],
                [f"alpha_{alpha}"],
                "E2B error vs tau",
                plots_dir / f"E2B_tau_alpha_{alpha}.png",
            )
        diag = rows[0].get("filtering_diagnostics", {})
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
    if name in {"E2A", "E3C", "E6A"}:
        curve = _median_rank_curve(rows)
        if curve is not None:
            _plot_curve(
                list(range(1, len(curve) + 1)),
                [curve],
                ["median_rank_curve"],
                f"{name} list quality",
                plots_dir / f"{name}_rank_curve.png",
            )
    if name in {"E4A", "E4B", "E4C", "E4D", "E4E"}:
        variants = sorted({row["variant"] for row in rows})
        errors = [np.median([row["oracle_param_error"] for row in rows if row["variant"] == v]) for v in variants]
        sel = [np.median([row["selected_pred_risk"] for row in rows if row["variant"] == v]) for v in variants]
        _plot_curve(variants, [errors, sel], ["oracle_error", "selected_risk"], f"{name} ablation", plots_dir / f"{name}_ablation.png")
    if name in {"E5A", "E5B", "E5C"}:
        if rows[0].get("contamination_stats"):
            outlier_fracs = rows[0]["contamination_stats"]["outlier_fraction"]
            _plot_hist(outlier_fracs, "Bucket contamination", plots_dir / "E5_contamination_hist.png")
        pruning = rows[0].get("pruning_stats")
        if pruning:
            _plot_curve(list(range(len(pruning["precision"]))), [pruning["precision"], pruning["recall"]], ["precision", "recall"], "Pruning PR", plots_dir / "E5_pr_curve.png")
        diag = rows[0].get("filtering_diagnostics", {})
        if diag.get("lambda_max"):
            _plot_curve(list(range(len(diag["lambda_max"]))), [diag["lambda_max"]], ["lambda_max"], "Potential decrease", plots_dir / "E5_potential.png")
    if name == "E6B":
        pca_points = rows[0].get("pca_points")
        pca_labels = rows[0].get("pca_labels")
        if pca_points is not None and pca_labels is not None:
            points = np.array(pca_points)
            labels = np.array(pca_labels)
            _plot_scatter(points, labels, "E6B PCA candidate scatter", plots_dir / "E6B_pca_scatter.png")
    if name == "E7A":
        pairs = sorted([(row["nnz"], row["timing_sketch"], row["timing_solve"]) for row in rows])
        nnz_vals = [p[0] for p in pairs]
        sketch_times = [p[1] for p in pairs]
        solve_times = [p[2] for p in pairs]
        _plot_curve(
            nnz_vals,
            [sketch_times, solve_times],
            ["sketch", "solve"],
            "E7A runtime scaling",
            plots_dir / "E7A_runtime.png",
        )
        if nnz_vals:
            ref_x = np.array(nnz_vals)
            ref_y = (ref_x / ref_x[0]) * sketch_times[0]
            _plot_curve(
                ref_x,
                [ref_y],
                ["linear_ref"],
                "E7A sketch reference",
                plots_dir / "E7A_sketch_ref.png",
            )
    if name == "E7B":
        timing_keys = ["timing_sketch", "timing_aggregation", "timing_solve"]
        averages = {
            key.replace("timing_", ""): float(np.mean([row[key] for row in rows if row[key] is not None]))
            for key in timing_keys
        }
        _plot_stacked_bar(["total"], {k: [v] for k, v in averages.items()}, "E7B time breakdown", plots_dir / "E7B_breakdown.png")
    if name == "E8A":
        variants = sorted({row["variant"] for row in rows})
        errors = [np.median([row["oracle_param_error"] for row in rows if row["variant"] == v]) for v in variants]
        _plot_curve(variants, [errors], ["oracle_error"], "E8A alpha misspec", plots_dir / "E8A_alpha.png")
    if name == "E8B":
        buckets = sorted({row["buckets"] for row in rows})
        reps = sorted({row["repetitions"] for row in rows})
        matrix = np.zeros((len(buckets), len(reps)))
        for i, b in enumerate(buckets):
            for j, r in enumerate(reps):
                subset = [row for row in rows if row["buckets"] == b and row["repetitions"] == r]
                if subset:
                    matrix[i, j] = np.median([row["oracle_param_error"] for row in subset])
        _plot_heatmap(matrix, "E8B sensitivity", plots_dir / "E8B_heatmap.png", reps, buckets)


def experiment_registry() -> Dict[str, List[ExperimentConfig]]:
    configs: Dict[str, List[ExperimentConfig]] = {}

    e1a = []
    for alpha in [0.05, 0.1, 0.2]:
        for d in [50, 100, 200]:
            n0 = int(np.ceil(2 * d / alpha))
            for n in [n0, 2 * n0, 4 * n0, 8 * n0, 16 * n0, 32 * n0]:
                e1a.append(
                    ExperimentConfig(
                        name="E1A",
                        variant=f"alpha_{alpha}_d_{d}",
                        data_type="synthetic",
                        config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                    )
                )
    configs["E1A"] = e1a

    e1b = []
    for alpha in [0.05, 0.1]:
        for c in [5, 10, 20]:
            for d in [25, 50, 100, 200, 400]:
                n = int(np.ceil(c * d / alpha))
                e1b.append(
                    ExperimentConfig(
                        name="E1B",
                        variant=f"alpha_{alpha}_c_{c}",
                        data_type="synthetic",
                        config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                    )
                )
    configs["E1B"] = e1b

    e1c = []
    d = 200
    alphas = [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    ratios = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96]
    for alpha in alphas:
        for ratio in ratios:
            n = int(np.ceil(ratio * d))
            e1c.append(
                ExperimentConfig(
                    name="E1C",
                    variant=f"alpha_{alpha}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=d,
                        alpha=alpha,
                        success_threshold_mult=2.0,
                    ),
                )
            )
    configs["E1C"] = e1c

    e2a = []
    for alpha in [0.05, 0.1, 0.2]:
        for mode in ["uniform", "gaussian", "sign_flip", "fake_model"]:
            n = int(np.ceil(20 * 200 / alpha))
            e2a.append(
                ExperimentConfig(
                    name="E2A",
                    variant=f"alpha_{alpha}_{mode}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        corruption_model="label_only",
                        label_corruption=mode,
                    ),
                )
            )
    configs["E2A"] = e2a

    e2b = []
    for alpha in [0.02, 0.05, 0.1, 0.2]:
        n = int(np.ceil(30 * 200 / alpha))
        for tau in [1, 10, 100, 1000]:
            e2b.append(
                ExperimentConfig(
                    name="E2B",
                    variant=f"alpha_{alpha}_tau_{tau}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        corruption_model="leverage",
                        leverage_tau=tau,
                    ),
                )
            )
    configs["E2B"] = e2b

    e2c = []
    for alpha in [0.05, 0.1, 0.2]:
        n = int(np.ceil(30 * 200 / alpha))
        for tau in [1, 100]:
            e2c.append(
                ExperimentConfig(
                    name="E2C",
                    variant=f"alpha_{alpha}_tau_{tau}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        corruption_model="adaptive",
                        leverage_tau=tau,
                    ),
                    baselines=["countsketch", "huber", "theilsen", "ransac", "lad", "tukey", "lts"],
                )
            )
    configs["E2C"] = e2c

    e2d = []
    for alpha in [0.05, 0.1]:
        n = int(np.ceil(40 * 200 / alpha))
        for nu in [2, 3, 5]:
            e2d.append(
                ExperimentConfig(
                    name="E2D",
                    variant=f"alpha_{alpha}_nu_{nu}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        corruption_model="heavy_tailed",
                        heavy_tail_nu=nu,
                    ),
                )
            )
    configs["E2D"] = e2d

    e3a = []
    for d in [8, 12, 16, 20]:
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            n = int(np.ceil(30 * d / alpha))
            e3a.append(
                ExperimentConfig(
                    name="E3A",
                    variant=f"d_{d}_alpha_{alpha}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=d,
                        alpha=alpha,
                        corruption_model="leverage",
                        leverage_tau=100,
                    ),
                    baselines=["sos_stub"],
                )
            )
    configs["E3A"] = e3a

    e3b = []
    for d in [100, 200]:
        for alpha in [0.05, 0.1, 0.2]:
            n = int(np.ceil(30 * d / alpha))
            for b in [16, 32, 64]:
                e3b.append(
                    ExperimentConfig(
                        name="E3B",
                        variant=f"d_{d}_alpha_{alpha}_b_{b}_unbatched",
                        data_type="synthetic",
                        config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                    )
                )
                e3b.append(
                    ExperimentConfig(
                        name="E3B",
                        variant=f"d_{d}_alpha_{alpha}_b_{b}",
                        data_type="synthetic",
                        config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                        extras={"batched": True, "batch_size": b},
                        baselines=["batched"],
                    )
                )
                e3b.append(
                    ExperimentConfig(
                        name="E3B",
                        variant=f"d_{d}_alpha_{alpha}_b_{b}_true",
                        data_type="synthetic",
                        config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                        extras={"fixed_batches": True, "batch_size": b},
                    )
                )
    configs["E3B"] = e3b

    e3c = []
    for alpha in [0.02, 0.05, 0.1, 0.2]:
        n = int(np.ceil(30 * 200 / alpha))
        for tau in [1, 100, 1000]:
            e3c.append(
                ExperimentConfig(
                    name="E3C",
                    variant=f"alpha_{alpha}_tau_{tau}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        corruption_model="leverage",
                        leverage_tau=tau,
                    ),
                    baselines=["huber", "theilsen", "ransac", "lad", "tukey", "lts"],
                )
            )
    configs["E3C"] = e3c

    e4a = []
    for sketch in ["expander", "one_hash", "countsketch"]:
        e4a.append(
            ExperimentConfig(
                name="E4A",
                variant=sketch,
                data_type="synthetic",
                config=SyntheticConfig(
                    n_train=6000,
                    n_test=2000,
                    d=200,
                    alpha=0.1,
                    corruption_model="leverage",
                    leverage_tau=100,
                    sketch_type=sketch,
                ),
            )
        )
    e4a.append(
        ExperimentConfig(
            name="E4A",
            variant="jl_ols",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
            ),
            baselines=["jl_ols"],
        )
    )
    configs["E4A"] = e4a

    e4b = []
    for method in ["mean", "mom", "geom_median"]:
        e4b.append(
            ExperimentConfig(
                name="E4B",
                variant=method,
                data_type="synthetic",
                config=SyntheticConfig(
                    n_train=6000,
                    n_test=2000,
                    d=200,
                    alpha=0.1,
                    corruption_model="leverage",
                    leverage_tau=100,
                    robust_method=method,
                ),
            )
        )
    configs["E4B"] = e4b

    configs["E4C"] = [
        ExperimentConfig(
            name="E4C",
            variant="filter_on",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
                enable_filtering=True,
            ),
        ),
        ExperimentConfig(
            name="E4C",
            variant="filter_off",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
                enable_filtering=False,
                filtering_rounds=0,
            ),
        ),
    ]

    configs["E4D"] = [
        ExperimentConfig(
            name="E4D",
            variant="signs_on",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
                use_signs=True,
            ),
        ),
        ExperimentConfig(
            name="E4D",
            variant="signs_off",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
                use_signs=False,
            ),
        ),
    ]

    e4e = []
    for seeds in [8, 16, 32, 64, 128]:
        e4e.append(
            ExperimentConfig(
                name="E4E",
                variant=f"seeds_{seeds}",
                data_type="synthetic",
                config=SyntheticConfig(
                    n_train=6000,
                    n_test=2000,
                    d=200,
                    alpha=0.1,
                    corruption_model="leverage",
                    leverage_tau=100,
                    seeds=seeds,
                ),
            )
        )
    configs["E4E"] = e4e

    configs["E5A"] = [
        ExperimentConfig(
            name="E5A",
            variant="contamination",
            data_type="synthetic",
            config=SyntheticConfig(
                n_train=6000,
                n_test=2000,
                d=200,
                alpha=0.1,
                corruption_model="leverage",
                leverage_tau=100,
            ),
        )
    ]
    configs["E5B"] = configs["E5A"]
    configs["E5C"] = configs["E5A"]

    datasets = [
        "YearPredictionMSD",
        "Online News Popularity",
        "Superconductivity",
        "Appliances Energy",
        "CT Slice Location",
        "E2006-tfidf",
    ]
    e6a = []
    for dataset in datasets:
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.4]:
            base_ns = [5000, 10000, 20000]
            if dataset == "YearPredictionMSD":
                base_ns += [50000, 100000, 200000]
            for n in base_ns:
                for corruption in ["label_only", "feature_label", "foreign"]:
                    e6a.append(
                        ExperimentConfig(
                            name="E6A",
                            variant=f"{dataset}_alpha_{alpha}_n_{n}_{corruption}",
                            data_type="real",
                            config=RealDataConfig(
                                dataset_name=dataset,
                                n_subsample=n,
                                alpha=alpha,
                                corruption_model=corruption,
                                foreign_dataset="Online News Popularity" if corruption == "foreign" else None,
                            ),
                        )
                    )
    configs["E6A"] = e6a

    configs["E6B"] = [
        ExperimentConfig(
            name="E6B",
            variant="OnlineNews_mixture",
            data_type="real",
            config=RealDataConfig(
                dataset_name="Online News Popularity",
                n_subsample=10000,
                alpha=0.3,
                mixture_alpha=0.3,
            ),
            extras={"mixture": True},
        ),
        ExperimentConfig(
            name="E6B",
            variant="YearPrediction_mixture",
            data_type="real",
            config=RealDataConfig(
                dataset_name="YearPredictionMSD",
                n_subsample=50000,
                alpha=0.3,
                mixture_alpha=0.3,
            ),
            extras={"mixture": True},
        ),
    ]

    e7a = []
    for nnz in [5, 10, 20, 40, 80, 160]:
        for n in [2000, 5000, 10000, 20000]:
            e7a.append(
                ExperimentConfig(
                    name="E7A",
                    variant=f"sparse_{nnz}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=150000,
                        alpha=0.1,
                        sparse_nnz_per_row=nnz,
                    ),
                )
            )
    e7a.append(
        ExperimentConfig(
            name="E7A",
            variant="E2006",
            data_type="real",
            config=RealDataConfig(
                dataset_name="E2006-tfidf",
                n_subsample=20000,
                alpha=0.1,
            ),
        )
    )
    configs["E7A"] = e7a

    e7b = []
    for d in [50, 100, 200, 400]:
        for alpha in [0.05, 0.1, 0.2]:
            n = int(np.ceil(50 * d / alpha))
            e7b.append(
                ExperimentConfig(
                    name="E7B",
                    variant=f"d_{d}_alpha_{alpha}",
                    data_type="synthetic",
                    config=SyntheticConfig(n_train=n, n_test=2000, d=d, alpha=alpha),
                )
            )
    configs["E7B"] = e7b

    e8a = []
    for alpha in [0.05, 0.1, 0.2]:
        n = int(np.ceil(30 * 200 / alpha))
        for alpha_hat in [alpha / 4, alpha / 2, alpha, 2 * alpha, 4 * alpha]:
            e8a.append(
                ExperimentConfig(
                    name="E8A",
                    variant=f"alpha_hat_{alpha_hat}",
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=200,
                        alpha=alpha,
                        alpha_hat=alpha_hat,
                        corruption_model="leverage",
                        leverage_tau=100,
                    ),
                )
            )
    configs["E8A"] = e8a

    e8b = []
    d = 200
    alpha = 0.1
    n = int(np.ceil(30 * d / alpha))
    for B_mult in [4, 8, 16, 32]:
        for r in [1, 2, 4, 8]:
            e8b.append(
                ExperimentConfig(
                    name="E8B",
                    variant=str(B_mult * d),
                    data_type="synthetic",
                    config=SyntheticConfig(
                        n_train=n,
                        n_test=2000,
                        d=d,
                        alpha=alpha,
                        buckets=B_mult * d,
                        repetitions=r,
                    ),
                )
            )
    configs["E8B"] = e8b

    return configs


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
