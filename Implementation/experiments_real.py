# experiments_real.py
"""
Real-data experiments with optional multi-seed averaging.

Per seed:
  1) Load CASP (inliers) and Concrete (outliers) datasets.
  2) Map both to a shared d_target-dimensional feature space:
       standardize -> polynomial features -> PCA.
  3) Hold out a clean CASP test set.
  4) Build a mixed training set with alpha inliers + (permuted) outliers.
  5) Evaluate baselines, Expander-1, and Expander-L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import expander_sketch_list_regression

# Data loaders
def load_casp(path: str = "Datasets/CASP/CASP.csv"):
    """Load the CASP dataset. First column is response, rest are features."""
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    return X, y


def load_concrete(path: str = "Datasets/concrete+compressive+strength/Concrete_Data.xls"):
    """Load the Concrete Compressive Strength dataset."""
    df = pd.read_excel(path)
    y = df.iloc[:, -1].to_numpy(dtype=float)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    return X, y

# Shared preprocessing to 10D
def joint_preprocess_to_10d(
    X_in: np.ndarray,
    X_out: np.ndarray,
    degree: int = 2,
    d_target: int = 10,
    random_state: int = 0,
):
    """
    Map inlier and outlier feature matrices to a shared d_target-dimensional
    space via: standardization -> polynomial expansion -> PCA.

    If the original feature dimensions differ, we pad the smaller one
    with zeros so that PCA sees a consistent dimension.
    """
    d_in = X_in.shape[1]
    d_out = X_out.shape[1]

    if d_in != d_out:
        d_max = max(d_in, d_out)
        if d_in < d_max:
            X_in = np.hstack([X_in, np.zeros((X_in.shape[0], d_max - d_in))])
        if d_out < d_max:
            X_out = np.hstack([X_out, np.zeros((X_out.shape[0], d_max - d_out))])

    X_all = np.vstack([X_in, X_out])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_std)

    pca = PCA(n_components=d_target, random_state=random_state)
    X_10d = pca.fit_transform(X_poly)

    n_in = X_in.shape[0]
    return X_10d[:n_in], X_10d[n_in:]


# Real-data mixture builder
def build_real_mix(
    X_in: np.ndarray,
    y_in: np.ndarray,
    X_out: np.ndarray,
    y_out: np.ndarray,
    alpha: float = 0.30,
    n_total: int = 1400,
    random_state: int = 0,
):
    """
    Build a synthetic mixture of CASP inliers and Concrete outliers.

    - Sample n_in = alpha * n_total inliers and n_out = n_total - n_in outliers.
    - Permute the outlier responses to destroy any alignment.
    - Return a shuffled design/response pair plus the inlier mask.
    """
    rng = np.random.default_rng(random_state)

    n_in_total = X_in.shape[0]
    n_out_total = X_out.shape[0]

    n_in = int(alpha * n_total)
    n_out = n_total - n_in

    if n_in > n_in_total:
        raise ValueError(f"Requested {n_in} inliers but only {n_in_total} available.")
    if n_out > n_out_total:
        raise ValueError(f"Requested {n_out} outliers but only {n_out_total} available.")

    idx_in = rng.choice(n_in_total, n_in, replace=False)
    idx_out = rng.choice(n_out_total, n_out, replace=False)

    X_in_sub = X_in[idx_in]
    y_in_sub = y_in[idx_in]

    X_out_sub = X_out[idx_out]
    y_out_sub = y_out[idx_out]

    # Permute outlier labels
    y_out_perm = rng.permutation(y_out_sub)

    X_mix = np.vstack([X_in_sub, X_out_sub])
    y_mix = np.concatenate([y_in_sub, y_out_perm])
    inlier_mask = np.array([True] * n_in + [False] * n_out, dtype=bool)

    perm = rng.permutation(n_total)
    return X_mix[perm], y_mix[perm], inlier_mask[perm]


# Aggregation over seeds
def aggregate_rows(rows: list[dict]) -> dict:
    """
    Aggregate a list of per-seed result dicts into mean/std for numeric keys.

    - Numeric keys -> "<key>_mean", "<key>_std".
    - Non-numeric keys are copied if identical across seeds, otherwise dropped.
    """
    if not rows:
        return {}

    agg: dict = {}
    first = rows[0]

    for k, v in first.items():
        vals = [r.get(k) for r in rows]

        # Numeric entries
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in vals):
            arr = np.array(vals, dtype=float)
            agg[f"{k}_mean"] = float(arr.mean())
            agg[f"{k}_std"] = float(arr.std())
        # Non-numeric but identical across seeds
        elif len(set(vals)) == 1:
            agg[k] = v

    agg["num_seeds"] = len(rows)
    return agg


# Single-seed run
def run_real_experiment(
    alpha: float = 0.30,
    n_total: int = 1400,
    preprocess_degree: int = 2,
    preprocess_d_target: int = 10,
    seed: int = 0,
) -> dict:
    """
    Run the real-data mixture experiment for a single random seed.
    Returns a dictionary of MSEs for all methods.
    """
    # 1. Load datasets
    X_casp_raw, y_casp = load_casp()
    X_conc_raw, y_conc = load_concrete()

    # 2. Preprocess to common d_target-dimensional space
    X_casp_10, X_conc_10 = joint_preprocess_to_10d(
        X_casp_raw,
        X_conc_raw,
        degree=preprocess_degree,
        d_target=preprocess_d_target,
        random_state=seed,
    )

    # 3. Build CASP-only clean test set
    rng = np.random.default_rng(123 + seed)
    n_test = 1000
    if n_test >= len(X_casp_10):
        raise ValueError("n_test must be smaller than the number of CASP samples.")
    idx_test = rng.choice(len(X_casp_10), size=n_test, replace=False)

    mask_test = np.zeros(len(X_casp_10), dtype=bool)
    mask_test[idx_test] = True

    X_test = X_casp_10[mask_test]
    y_test = y_casp[mask_test]

    X_casp_train = X_casp_10[~mask_test]
    y_casp_train = y_casp[~mask_test]

    # 4. Oracle CASP-only regressor
    ols_oracle = LinearRegression().fit(X_casp_train, y_casp_train)
    mse_oracle = mean_squared_error(y_test, ols_oracle.predict(X_test))
    print(f"Oracle OLS test MSE = {mse_oracle:.4f}")

    # 5. Build mixture CASP-inliers + Concrete-outliers
    X_mix, y_mix, inlier_mask = build_real_mix(
        X_in=X_casp_train,
        y_in=y_casp_train,
        X_out=X_conc_10,
        y_out=y_conc,
        alpha=alpha,
        n_total=n_total,
        random_state=seed,
    )

    eff_alpha = float(inlier_mask.mean())

    # 6. Baseline regressors
    mse_ols = mean_squared_error(
        y_test, LinearRegression().fit(X_mix, y_mix).predict(X_test)
    )
    mse_ridge = mean_squared_error(y_test, X_test @ fit_ridge(X_mix, y_mix, alpha=1.0))
    mse_huber = mean_squared_error(y_test, X_test @ fit_huber(X_mix, y_mix))
    mse_ts = mean_squared_error(y_test, X_test @ fit_theilsen(X_mix, y_mix))
    mse_ransac = mean_squared_error(y_test, X_test @ fit_ransac(X_mix, y_mix))

    print(f"OLS mixture  MSE = {mse_ols:.4f}")
    print(f"Ridge        MSE = {mse_ridge:.4f}")
    print(f"Huber        MSE = {mse_huber:.4f}")
    print(f"Theil–Sen    MSE = {mse_ts:.4f}")
    print(f"RANSAC       MSE = {mse_ransac:.4f}")

    # 7. Expander-1
    beta1 = expander_sketch_regression_single_seed(
        X_mix,
        y_mix,
        alpha=eff_alpha,
        B=1000,
        r=8,
        dL=2,
        lambda_reg=1e-3,
        random_state=seed,
    )
    mse_exp1 = mean_squared_error(y_test, X_test @ beta1)

    # 8. Expander-L
    candidates = expander_sketch_list_regression(
        X_mix,
        y_mix,
        alpha=eff_alpha,
        r=8,
        B=1000,
        dL=2,
        T=7,
        R=10,
        lambda_reg=1e-3,
        theta=0.10,
        rho=0.45,
        cluster_radius=0.0,
        random_state=seed,
    )

    mse_expL = min(mean_squared_error(y_test, X_test @ c) for c in candidates)

    return {
        "seed": seed,
        "alpha": alpha,
        "n_total": n_total,
        "eff_alpha": eff_alpha,
        "mse_oracle": mse_oracle,
        "mse_ols": mse_ols,
        "mse_ridge": mse_ridge,
        "mse_huber": mse_huber,
        "mse_theilsen": mse_ts,
        "mse_ransac": mse_ransac,
        "mse_exp1": mse_exp1,
        "mse_expL": mse_expL,
    }


# Driver
def main(seeds=(0,)):
    per_seed = []

    # Pretty formatting helper
    def pr(method, value):
        print(f"    {method:<12}: {value:>10.4f}")

    for s in seeds:
        res = run_real_experiment(seed=s)
        per_seed.append(res)

        print(f"\n[seed={s}]  eff_alpha = {res['eff_alpha']:.3f}")
        pr("OLS",        res["mse_ols"])
        pr("Ridge",      res["mse_ridge"])
        pr("Huber",      res["mse_huber"])
        pr("Theil–Sen",  res["mse_theilsen"])
        pr("RANSAC",     res["mse_ransac"])
        pr("Expander-1", res["mse_exp1"])
        pr("Expander-L", res["mse_expL"])

    # Aggregate over seeds
    if len(per_seed) > 1:
        agg = aggregate_rows(per_seed)

        print(f"\n=== Aggregate over seeds ({agg['num_seeds']} runs) ===")

        def pr_agg(key, ag):
            mean_k = f"{key}_mean"
            std_k  = f"{key}_std"
            if mean_k in ag:
                print(f"    {key:<12}: {ag[mean_k]:>10.4f}  (± {ag[std_k]:.4f})")

        for key in [
            "mse_oracle",
            "mse_ols",
            "mse_ridge",
            "mse_huber",
            "mse_theilsen",
            "mse_ransac",
            "mse_exp1",
            "mse_expL",
        ]:
            pr_agg(key, agg)


if __name__ == "__main__":
    # Change seeds to a list (e.g., [0, 1, 2, 3, 4]) for multi-seed averaging.
    main(seeds=[0, 1, 2, 3, 4])
