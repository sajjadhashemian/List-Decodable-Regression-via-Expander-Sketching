"""
Core utilities for synthetic experiments:
  - CSV logging helpers
  - Expander graph construction
  - Single- and multi-seed experiment runners
"""

import os
import csv
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data import generate_regression_with_outliers
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import expander_sketch_list_regression
from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen


def append_result_wide(csv_path, result_dict):
    """
    Append a single 'wide' result row to csv_path.

    The row will contain all keys of result_dict as columns.
    If the file doesn't exist, a header row is written first.

    By default, CSVs are stored under the 'Tables/' directory,
    mirroring how plots are stored under 'Figures/'.
    """
    tables_dir = "Tables"
    os.makedirs(tables_dir, exist_ok=True)

    if not os.path.dirname(csv_path):
        csv_path = os.path.join(tables_dir, csv_path)

    keys = list(result_dict.keys())
    if "alpha" in keys:
        keys.remove("alpha")
        fieldnames = ["alpha"] + keys
    else:
        fieldnames = keys

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result_dict.get(k, "") for k in fieldnames})


def make_random_regular_bipartite_expander(n, B, dL, seed=0):
    """
    Build a random left-regular bipartite graph using NetworkX:

        - Left nodes:  0 .. n-1      (samples)
        - Right nodes: n .. n+B-1    (buckets)

    Each left node has degree exactly dL.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + B), bipartite=1)

    for i in range(n):
        neighbors = rng.choice(np.arange(n, n + B), size=dL, replace=False)
        for v in neighbors:
            G.add_edge(i, v)

    return G


def _exp_single_params(exp_single_dL, exp_single_lambda):
    return {
        "exp_single_dL": exp_single_dL,
        "exp_single_lambda": exp_single_lambda,
    }


def _exp_list_params(
    exp_list_T,
    exp_list_R,
    exp_list_lambda,
    exp_list_dL=None,
    exp_list_r=None,
    exp_list_theta=None,
    exp_list_rho=None,
    exp_list_cluster_radius=None,
):
    params = {
        "exp_list_T": exp_list_T,
        "exp_list_R": exp_list_R,
        "exp_list_lambda": exp_list_lambda,
    }
    if exp_list_dL is not None:
        params["exp_list_dL"] = exp_list_dL
    if exp_list_r is not None:
        params["exp_list_r"] = exp_list_r
    if exp_list_theta is not None:
        params["exp_list_theta"] = exp_list_theta
    if exp_list_rho is not None:
        params["exp_list_rho"] = exp_list_rho
    if exp_list_cluster_radius is not None:
        params["exp_list_cluster_radius"] = exp_list_cluster_radius
    return params


_COMMON_META_KEYS = {"alpha", "mode", "use_networkx", "n", "d", "outlier_scale"}
_EXP_SINGLE_META_KEYS = {"exp_single_dL", "exp_single_lambda"}
_EXP_LIST_META_KEYS_MIN = {"exp_list_T", "exp_list_R", "exp_list_lambda"}
_EXP_LIST_META_KEYS_FULL = _EXP_LIST_META_KEYS_MIN | {
    "exp_list_dL",
    "exp_list_r",
    "exp_list_theta",
    "exp_list_rho",
    "exp_list_cluster_radius",
}


def aggregate_multi_seed(per_seed_results, meta_keys):
    """
    Given a list of per-seed result dicts, return a dict with:
      - meta_keys copied as-is (alpha, mode, etc.)
      - numeric metrics summarized with mean/std suffixes
      - seed marker ("mean") and num_seeds
    """
    aggregated = {}
    if not per_seed_results:
        return aggregated

    first = per_seed_results[0]
    for k in meta_keys:
        if k in first:
            aggregated[k] = first[k]

    aggregated["seed"] = "mean"
    aggregated["num_seeds"] = len(per_seed_results)

    for key in first.keys():
        if key in meta_keys or key == "seed":
            continue
        vals = [res.get(key) for res in per_seed_results]
        numeric_vals = [
            v
            for v in vals
            if isinstance(v, (int, float, np.integer, np.floating))
        ]
        if numeric_vals and len(numeric_vals) == len(vals):
            mean_v = float(np.mean(numeric_vals))
            std_v = float(np.std(numeric_vals))
            aggregated[f"{key}_mean"] = mean_v
            aggregated[f"{key}_std"] = std_v
        elif len(set(vals)) == 1:
            aggregated[key] = vals[0]
        else:
            aggregated[f"{key}_mean"] = float(np.nan)
            aggregated[f"{key}_std"] = float(np.nan)

    return aggregated


def run_multi_seed(
    single_seed_fn,
    seeds,
    meta_keys,
    base_kwargs,
    seed_kw: str = "random_state",
):
    """
    Generic helper:
      - single_seed_fn: function that runs one experiment (takes a seed via seed_kw)
      - seeds: iterable of seeds
      - meta_keys: keys to carry over unchanged when aggregating
      - base_kwargs: kwargs passed to single_seed_fn (same for all seeds)
      - seed_kw: name of the kwarg for the seed (default: 'random_state')

    Returns (aggregated_results, per_seed_results).
    """
    per_seed_results = []
    for seed in seeds:
        kwargs = dict(base_kwargs)
        kwargs[seed_kw] = seed
        res = single_seed_fn(**kwargs)
        res["seed"] = seed
        per_seed_results.append(res)

    aggregated = aggregate_multi_seed(per_seed_results, meta_keys)
    return aggregated, per_seed_results


def run_all_methods_for_alpha(
    alpha,
    outlier_mode: str = "uniform",
    random_state: int = 0,
    use_networkx_expander: bool = False,
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    # Expander hyperparameters (exposed for sweeps)
    exp_single_dL: int = 2,
    exp_single_lambda: float = 1e-3,
    exp_list_T: int = 7,
    exp_list_R: int = 10,
    exp_list_lambda: float = 1e-3,
):
    """
    Run one synthetic experiment for a given alpha and outlier model.
    Returns a dict of metrics for all methods.
    """
    # Expander-single parameters
    B_sketch = 1000
    exp_single_r = 8

    # Expander-list parameters (Algorithm 1)
    exp_list_r = 8
    exp_list_theta = 0.1
    exp_list_rho = 0.5
    exp_list_cluster_radius = 0.0

    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    if use_networkx_expander:
        G_expander = make_random_regular_bipartite_expander(
            n=n,
            B=B_sketch,
            dL=exp_single_dL,
            seed=random_state,
        )
    else:
        G_expander = None

    rng = np.random.default_rng(123 + random_state)
    X_test = rng.normal(size=(2000, d))
    y_test = X_test @ w_star

    results = {
        "alpha": alpha,
        "mode": outlier_mode,
        "use_networkx": int(use_networkx_expander),
        "n": n,
        "d": d,
        "outlier_scale": outlier_scale,
        **_exp_single_params(exp_single_dL, exp_single_lambda),
        **_exp_list_params(exp_list_T, exp_list_R, exp_list_lambda),
    }

    # Baselines: OLS
    ols = LinearRegression().fit(X, y)
    beta_ols = ols.coef_
    results["ols_err"] = np.linalg.norm(beta_ols - w_star)
    results["ols_mse"] = mean_squared_error(y_test, X_test @ beta_ols)

    # Sklearn baselines
    beta_ridge = fit_ridge(X, y, alpha=1.0)
    results["ridge_err"] = np.linalg.norm(beta_ridge - w_star)
    results["ridge_mse"] = mean_squared_error(y_test, X_test @ beta_ridge)

    beta_huber = fit_huber(X, y, alpha=0.0001, epsilon=1.35)
    results["huber_err"] = np.linalg.norm(beta_huber - w_star)
    results["huber_mse"] = mean_squared_error(y_test, X_test @ beta_huber)

    beta_ransac = fit_ransac(
        X,
        y,
        min_samples=None,
        residual_threshold=None,
        max_trials=100,
    )
    results["ransac_err"] = np.linalg.norm(beta_ransac - w_star)
    results["ransac_mse"] = mean_squared_error(y_test, X_test @ beta_ransac)

    beta_ts = fit_theilsen(X, y)
    results["theilsen_err"] = np.linalg.norm(beta_ts - w_star)
    results["theilsen_mse"] = mean_squared_error(y_test, X_test @ beta_ts)

    # Expander-single (1 seed)
    beta_exp_single = expander_sketch_regression_single_seed(
        X,
        y,
        alpha=alpha,
        B=B_sketch,
        r=exp_single_r,
        dL=exp_single_dL,
        lambda_reg=exp_single_lambda,
        random_state=random_state,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )
    results["exp_single_err"] = np.linalg.norm(beta_exp_single - w_star)
    results["exp_single_mse"] = mean_squared_error(
        y_test,
        X_test @ beta_exp_single,
    )

    # Expander-list (Algorithm 1)
    candidates = expander_sketch_list_regression(
        X,
        y,
        alpha=alpha,
        r=exp_list_r,
        B=B_sketch,
        dL=exp_single_dL,
        T=exp_list_T,
        R=exp_list_R,
        lambda_reg=exp_list_lambda,
        theta=exp_list_theta,
        rho=exp_list_rho,
        cluster_radius=exp_list_cluster_radius,
        random_state=random_state,
        verbose=False,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )

    best_err = None
    best_mse = None
    for beta in candidates:
        err = np.linalg.norm(beta - w_star)
        mse = mean_squared_error(y_test, X_test @ beta)
        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    results["exp_list_num_cands"] = len(candidates)
    results["exp_list_best_err"] = best_err
    results["exp_list_best_mse"] = best_mse

    return results


def run_all_methods_for_alpha_multi_seed(
    alpha,
    outlier_mode: str = "uniform",
    seeds = (0, 1, 2, 3, 4),
    use_networkx_expander: bool = False,
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    exp_single_dL: int = 2,
    exp_single_lambda: float = 1e-3,
    exp_list_T: int = 7,
    exp_list_R: int = 10,
    exp_list_lambda: float = 1e-3,
):
    """
    Run run_all_methods_for_alpha over multiple seeds and aggregate metrics.
    Returns (aggregated_results, per_seed_results).
    """
    meta_keys = _COMMON_META_KEYS | _EXP_SINGLE_META_KEYS | _EXP_LIST_META_KEYS_MIN

    base_kwargs = dict(
        alpha=alpha,
        outlier_mode=outlier_mode,
        use_networkx_expander=use_networkx_expander,
        n=n,
        d=d,
        outlier_scale=outlier_scale,
        **_exp_single_params(exp_single_dL, exp_single_lambda),
        **_exp_list_params(exp_list_T, exp_list_R, exp_list_lambda),
    )

    return run_multi_seed(
        single_seed_fn=run_all_methods_for_alpha,
        seeds=seeds,
        meta_keys=meta_keys,
        base_kwargs=base_kwargs,
        seed_kw="random_state",
    )


def run_expander_list_for_alpha(
    alpha,
    outlier_mode: str = "uniform",
    random_state: int = 0,
    use_networkx_expander: bool = False,
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    # Expander-L hyperparameters (for sweeps)
    exp_list_T: int = 7,
    exp_list_R: int = 10,
    exp_list_lambda: float = 1e-3,
    exp_list_dL: int = 2,
    exp_list_r: int = 8,
    exp_list_theta: float = 0.1,
    exp_list_rho: float = 0.5,
    exp_list_cluster_radius: float = 0.0,
    B_sketch: int = 1000,
):
    """
    Run a synthetic experiment for a given alpha and outlier model,
    but ONLY for Expander-L (no baselines, no Expander-1).
    """
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    if use_networkx_expander:
        G_expander = make_random_regular_bipartite_expander(
            n=n,
            B=B_sketch,
            dL=exp_list_dL,
            seed=random_state,
        )
    else:
        G_expander = None

    rng = np.random.default_rng(123 + random_state)
    X_test = rng.normal(size=(2000, d))
    y_test = X_test @ w_star

    results = {
        "alpha": alpha,
        "mode": outlier_mode,
        "use_networkx": int(use_networkx_expander),
        "n": n,
        "d": d,
        "outlier_scale": outlier_scale,
        **_exp_list_params(
            exp_list_T,
            exp_list_R,
            exp_list_lambda,
            exp_list_dL,
            exp_list_r,
            exp_list_theta,
            exp_list_rho,
            exp_list_cluster_radius,
        ),
    }

    candidates = expander_sketch_list_regression(
        X,
        y,
        alpha=alpha,
        r=exp_list_r,
        B=B_sketch,
        dL=exp_list_dL,
        T=exp_list_T,
        R=exp_list_R,
        lambda_reg=exp_list_lambda,
        theta=exp_list_theta,
        rho=exp_list_rho,
        cluster_radius=exp_list_cluster_radius,
        random_state=random_state,
        verbose=False,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )

    best_err = None
    best_mse = None
    for beta in candidates:
        err = np.linalg.norm(beta - w_star)
        mse = mean_squared_error(y_test, X_test @ beta)
        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    results["exp_list_num_cands"] = len(candidates)
    results["exp_list_best_err"] = best_err
    results["exp_list_best_mse"] = best_mse

    return results


def run_expander_list_for_alpha_multi_seed(
    alpha,
    outlier_mode: str = "uniform",
    seeds = (0, 1, 2, 3, 4),
    use_networkx_expander: bool = False,
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    # Expander-L hyperparameters (for sweeps)
    exp_list_T: int = 7,
    exp_list_R: int = 10,
    exp_list_lambda: float = 1e-3,
    exp_list_dL: int = 2,
    exp_list_r: int = 8,
    exp_list_theta: float = 0.1,
    exp_list_rho: float = 0.5,
    exp_list_cluster_radius: float = 0.0,
    B_sketch: int = 1000,
):
    """
    Run run_expander_list_for_alpha over multiple seeds and aggregate metrics.
    Returns (aggregated_results, per_seed_results).
    """
    meta_keys = _COMMON_META_KEYS | _EXP_LIST_META_KEYS_FULL

    base_kwargs = dict(
        alpha=alpha,
        outlier_mode=outlier_mode,
        use_networkx_expander=use_networkx_expander,
        n=n,
        d=d,
        outlier_scale=outlier_scale,
        **_exp_list_params(
            exp_list_T,
            exp_list_R,
            exp_list_lambda,
            exp_list_dL,
            exp_list_r,
            exp_list_theta,
            exp_list_rho,
            exp_list_cluster_radius,
        ),
        B_sketch=B_sketch,
    )

    return run_multi_seed(
        single_seed_fn=run_expander_list_for_alpha,
        seeds=seeds,
        meta_keys=meta_keys,
        base_kwargs=base_kwargs,
        seed_kw="random_state",
    )
