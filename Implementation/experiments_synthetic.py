# sanity_checks.py
import os
import csv
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from data import generate_regression_with_outliers
from expander_sketch_regression import bucketed_median_regression
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import (
    expander_sketch_list_regression,
    oracle_inlier_bucket_regression,
    # debug_bucket_contamination,
    # debug_bucket_scores_vs_goodness,
    # debug_survivor_goodness,
)
from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen

def append_result_wide(csv_path, result_dict):
    """
    Append a single 'wide' result row to csv_path.

    The row will contain all keys of result_dict as columns.
    If the file doesn't exist, a header row is written first.
    """
    # Make sure alpha is first column in a nice order
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

def run_all_methods_for_alpha(alpha, outlier_mode="uniform", random_state=0):
    """
    Run one synthetic experiment for a given alpha and outlier model.
    Returns a dict of metrics for all methods.

    This function is the core per-alpha experiment used by:
    - save_uniform_results.py
    - plot_uniform.py
    - future experiment sweep scripts
    """

    # 0. Global experiment knobs (easy to tune later)
    n = 5000
    d = 20
    noise_std = 0.1
    outlier_scale = 10.0

    # Oracle bucket parameters
    r_oracle = 5
    B_oracle = 1000
    k_in_min = 5
    k_out_max = 5

    # MoM (batch) parameters
    mom_n_buckets = 50
    mom_n_reps = 5
    mom_lambda = 1e-3

    # Expander-single parameters
    B_sketch = 1000
    exp_single_r = 8
    exp_single_dL = 2
    exp_single_lambda = 1e-4

    # Expander-list parameters (Algorithm 1)
    exp_list_r = 8
    exp_list_T = 7
    exp_list_R = 10
    exp_list_lambda = 1e-3
    exp_list_theta = 0.1
    exp_list_rho = 0.5
    exp_list_cluster_radius = 0.0

    # Decide pruning rule:
    prune_mode = "flip" if outlier_mode == "signflip" else "paper"

    # 1. Generate synthetic data
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        # current generate_regression_with_outliers does not take noise_std
        random_state=random_state,
    )

    # Test set for all methods
    rng = np.random.default_rng(123 + random_state)
    X_test = rng.normal(size=(2000, d))
    y_test = X_test @ w_star

    results = {
        "alpha": alpha,
        "mode": outlier_mode,
    }

    # 2. Baselines: OLS
    ols = LinearRegression().fit(X, y)
    beta_ols = ols.coef_
    results["ols_err"] = np.linalg.norm(beta_ols - w_star)
    results["ols_mse"] = mean_squared_error(y_test, X_test @ beta_ols)

    # 3. Oracle baseline
    if alpha < 1.0:
        beta_oracle, K_good, B_val = oracle_inlier_bucket_regression(
            X, y, inlier_mask=inlier_mask,
            alpha=alpha,
            r=r_oracle,
            B=B_oracle,
            dL=2,
            k_in_min=k_in_min,
            k_out_max=k_out_max,
            lambda_reg=1e-3,
            random_state=123,
        )
        results["oracle_err"] = np.linalg.norm(beta_oracle - w_star)
        results["oracle_mse"] = mean_squared_error(y_test, X_test @ beta_oracle)
        results["oracle_K_good"] = K_good
        results["oracle_total_buckets"] = B_val * r_oracle
    else:
        results["oracle_err"] = None
        results["oracle_mse"] = None

    # 4. Sklearn baselines
    # Ridge
    beta_ridge = fit_ridge(X, y, alpha=1.0)
    results["ridge_err"] = np.linalg.norm(beta_ridge - w_star)
    results["ridge_mse"] = mean_squared_error(y_test, X_test @ beta_ridge)

    # Huber
    beta_huber = fit_huber(X, y, alpha=0.0001, epsilon=1.35)
    results["huber_err"] = np.linalg.norm(beta_huber - w_star)
    results["huber_mse"] = mean_squared_error(y_test, X_test @ beta_huber)

    # RANSAC
    beta_ransac = fit_ransac(X, y, min_samples=None,
                             residual_threshold=None, max_trials=100)
    results["ransac_err"] = np.linalg.norm(beta_ransac - w_star)
    results["ransac_mse"] = mean_squared_error(y_test, X_test @ beta_ransac)

    # Theil–Sen
    beta_ts = fit_theilsen(X, y)
    results["theilsen_err"] = np.linalg.norm(beta_ts - w_star)
    results["theilsen_mse"] = mean_squared_error(y_test, X_test @ beta_ts)

    # 5. MoM (batch)
    beta_mom = bucketed_median_regression(
        X, y,
        n_buckets=mom_n_buckets,
        n_reps=mom_n_reps,
        lambda_reg=mom_lambda,
        random_state=42,
    )
    results["mom_err"] = np.linalg.norm(beta_mom - w_star)
    results["mom_mse"] = mean_squared_error(y_test, X_test @ beta_mom)

    # 6. Expander-single (1 seed)
    beta_exp_single = expander_sketch_regression_single_seed(
        X, y,
        alpha=alpha,
        B=B_sketch,
        r=exp_single_r,
        dL=exp_single_dL,
        lambda_reg=exp_single_lambda,
        random_state=123,
    )
    results["exp_single_err"] = np.linalg.norm(beta_exp_single - w_star)
    results["exp_single_mse"] = mean_squared_error(y_test, X_test @ beta_exp_single)


    # 7. Expander-list (Algorithm 1)
    candidates = expander_sketch_list_regression(
        X, y,
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
        random_state=123,
        prune_mode=prune_mode,
        verbose=False,
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

def check_ols_recovery(alpha_no_outliers=1.0, alpha_many_outliers=0.3):
    # Case 1: no outliers
    X1, y1, w_star1, mask1, info1 = generate_regression_with_outliers(
        n=5000, d=20,
        alpha=alpha_no_outliers,
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=0
    )
    ols1 = LinearRegression().fit(X1, y1)
    err1 = np.linalg.norm(ols1.coef_ - w_star1)
    print(f"[alpha={alpha_no_outliers}] OLS ||w_hat - w_star||_2 = {err1:.4f}")
    
    # Expander-list on clean data
    candidates_clean = expander_sketch_list_regression(
        X1, y1,
        alpha=alpha_no_outliers,
        r=5,
        B=None,
        dL=2,
        T=1,
        R=5,
        lambda_reg=1e-3,
        cluster_radius=0.0,
        random_state=123,
    )

    print(f"[alpha={alpha_no_outliers}] Expander-list (clean) produced {len(candidates_clean)} candidates")
    for idx, beta in enumerate(candidates_clean):
        err = np.linalg.norm(beta - w_star1)
        print(f"  Clean cand {idx}: ||w_hat - w_star||_2 = {err:.4f}")
    
    # Case 2: many outliers
    X2, y2, w_star2, mask2, info2 = generate_regression_with_outliers(
        n=5000, d=20, alpha=alpha_many_outliers, 
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=1
    )
    ols2 = LinearRegression().fit(X2, y2)
    err2 = np.linalg.norm(ols2.coef_ - w_star2)
    print(f"[alpha={alpha_many_outliers}] OLS ||w_hat - w_star||_2 = {err2:.4f}")

    # ----- Make test set ONCE here so all methods (including oracle) can use it -----
    rng = np.random.default_rng(123)
    X_test = rng.normal(size=(2000, X1.shape[1]))
    y_test1 = X_test @ w_star1
    y_test2 = X_test @ w_star2

    # --- Oracle check: how well can we do using only 'good' buckets? ---
    r_oracle = 5
    B_oracle = 1000

    beta_oracle, K_good, B_val = oracle_inlier_bucket_regression(
        X2,
        y2,
        inlier_mask=mask2,
        alpha=alpha_many_outliers,
        r=r_oracle,
        B=B_oracle,
        dL=2,
        k_in_min=5,
        k_out_max=5,
        lambda_reg=1e-3,
        random_state=123,
    )

    err_oracle = np.linalg.norm(beta_oracle - w_star2)
    y_pred_oracle = X_test @ beta_oracle
    mse_oracle = mean_squared_error(y_test2, y_pred_oracle)

    print(f"[alpha={alpha_many_outliers}] ORACLE (good-buckets-only) used {K_good} buckets out of total {B_oracle * r_oracle}")
    print(f"[alpha={alpha_many_outliers}] ORACLE ||w_hat - w_star||_2 = {err_oracle:.4f}")
    print(f"[alpha={alpha_many_outliers}] ORACLE test MSE = {mse_oracle:.4f}")

    # print("\n--- Debug: bucket contamination stats ---")
    # debug_bucket_contamination(
    #     X2,
    #     y2,
    #     inlier_mask=mask2,
    #     alpha=alpha_many_outliers,
    #     r=5,
    #     B=B_oracle,
    #     dL=2,
    #     R=1,
    #     random_state=123,
    # )

    # print("\n--- Debug: bucket scores vs goodness ---")
    # debug_bucket_scores_vs_goodness(
    #     X2,
    #     y2,
    #     inlier_mask=mask2,
    #     alpha=alpha_many_outliers,
    #     r=5,
    #     B=1000,
    #     dL=2,
    #     lambda_reg=1e-3,
    #     random_state=123,
    #     k_in_min=5,
    #     k_out_max=5,
    # )

    # print("\n--- Debug: survivor goodness (flip mode) ---")
    # debug_survivor_goodness(
    #     X2,
    #     y2,
    #     inlier_mask=mask2,
    #     alpha=alpha_many_outliers,
    #     r=8,
    #     B=1000,
    #     dL=2,
    #     T=5,
    #     lambda_reg=1e-3,
    #     eta=0.1,
    #     rho=0.4,
    #     random_state=123,
    #     prune_mode="flip",
    #     k_in_min=5,
    #     k_out_max=5,
    # )


    # OLS test MSEs
    mse1 = mean_squared_error(y_test1, X_test @ ols1.coef_)
    mse2 = mean_squared_error(y_test2, X_test @ ols2.coef_)

    print(f"[alpha={alpha_no_outliers}] OLS test MSE = {mse1:.4f}")
    print(f"[alpha={alpha_many_outliers}] OLS test MSE = {mse2:.4f}")

    # === Sklearn robust baselines on the high-outlier case ===

    # Ridge regression
    beta_ridge = fit_ridge(X2, y2, alpha=1.0)
    err_ridge = np.linalg.norm(beta_ridge - w_star2)
    mse_ridge = mean_squared_error(y_test2, X_test @ beta_ridge)
    print(f"[alpha={alpha_many_outliers}] Ridge ||w_hat - w_star||_2 = {err_ridge:.4f}")
    print(f"[alpha={alpha_many_outliers}] Ridge test MSE = {mse_ridge:.4f}")

    # Huber regression
    beta_huber = fit_huber(X2, y2, alpha=0.0001, epsilon=1.35)
    err_huber = np.linalg.norm(beta_huber - w_star2)
    mse_huber = mean_squared_error(y_test2, X_test @ beta_huber)
    print(f"[alpha={alpha_many_outliers}] Huber ||w_hat - w_star||_2 = {err_huber:.4f}")
    print(f"[alpha={alpha_many_outliers}] Huber test MSE = {mse_huber:.4f}")

    # RANSAC regression
    beta_ransac = fit_ransac(X2, y2, min_samples=None, residual_threshold=None, max_trials=100)
    err_ransac = np.linalg.norm(beta_ransac - w_star2)
    mse_ransac = mean_squared_error(y_test2, X_test @ beta_ransac)
    print(f"[alpha={alpha_many_outliers}] RANSAC ||w_hat - w_star||_2 = {err_ransac:.4f}")
    print(f"[alpha={alpha_many_outliers}] RANSAC test MSE = {mse_ransac:.4f}")

    # Theil-Sen regression
    beta_ts = fit_theilsen(X2, y2)
    err_ts = np.linalg.norm(beta_ts - w_star2)
    mse_ts = mean_squared_error(y_test2, X_test @ beta_ts)
    print(f"[alpha={alpha_many_outliers}] Theil-Sen ||w_hat - w_star||_2 = {err_ts:.4f}")
    print(f"[alpha={alpha_many_outliers}] Theil-Sen test MSE = {mse_ts:.4f}")

    # --- Test bucketed median regression on the high-outlier case ---
    beta_mom = bucketed_median_regression(X2, y2, n_buckets=50, n_reps=5, lambda_reg=1e-3, random_state=42)
    err_mom = np.linalg.norm(beta_mom - w_star2)

    y_pred_mom = X_test @ beta_mom
    mse_mom = mean_squared_error(y_test2, y_pred_mom)

    print(f"[alpha={alpha_many_outliers}] Bucketed-Median ||w_hat - w_star||_2 = {err_mom:.4f}")
    print(f"[alpha={alpha_many_outliers}] Bucketed-Median test MSE = {mse_mom:.4f}")
    
    # --- Test simplified expander-sketch regression on the high-outlier case ---
    beta_exp = expander_sketch_regression_single_seed(
        X2, y2,
        alpha=alpha_many_outliers,
        B=None,          # will pick ~ d/alpha
        r=8,             # more repetitions
        dL=3,            # each point hits 3 buckets
        lambda_reg=1e-4, # a bit less regularization
        random_state=123,
    )

    err_exp = np.linalg.norm(beta_exp - w_star2)

    y_pred_exp = X_test @ beta_exp
    mse_exp = mean_squared_error(y_test2, y_pred_exp)

    print(f"[alpha={alpha_many_outliers}] Expander-single ||w_hat - w_star||_2 = {err_exp:.4f}")
    print(f"[alpha={alpha_many_outliers}] Expander-single test MSE = {mse_exp:.4f}")

    # --- Multi-seed expander-sketch list-decoding (Algorithm 1 style) ---
    candidates = expander_sketch_list_regression(
    X2, y2,
    alpha=alpha_many_outliers,
    r=8,            # more repetitions
    B=B_oracle,     # 1000
    dL=2,
    T=7,            # more filtering rounds
    R=10,
    lambda_reg=1e-3,
    theta=0.1,        # stricter spectral test
    rho=0.5,        # prune more aggressively
    cluster_radius=0,
    random_state=123,
    prune_mode="paper",  # Algorithm 1
)


    print(f"[alpha={alpha_many_outliers}] Expander-list produced {len(candidates)} candidates")

    # Evaluate each candidate; find best one (this imitates the 'list-decoding' success event)
    best_err = None
    best_mse = None
    for idx, beta in enumerate(candidates):
        err = np.linalg.norm(beta - w_star2)

        y_pred = X_test @ beta
        mse = mean_squared_error(y_test2, y_pred)

        print(f"  Candidate {idx}: ||w_hat - w_star||_2 = {err:.4f}, test MSE = {mse:.4f}")

        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    if best_err is not None:
        print(f"  Best candidate: param error = {best_err:.4f}, test MSE = {best_mse:.4f}")

def sweep_alpha_uniform():
    """
    Run run_single_experiment for several alpha values under uniform outliers,
    print a compact summary, and return a list of result dicts.
    """
    alphas = [1.0, 0.7, 0.5, 0.3, 0.2]
    all_results = []

    for a in alphas:
        res = run_all_methods_for_alpha(a, outlier_mode="uniform", random_state=0)
        all_results.append(res)

        # --- pretty print to console ---
        print("\n===========================================")
        print(f"Results for alpha = {a:.2f}, outlier_mode = uniform")
        print("-------------------------------------------")
        print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
        if res.get("oracle_err") is not None:
            print(f"Oracle     : err = {res['oracle_err']:.4f},  mse = {res['oracle_mse']:.4f}")
        print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
        print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
        print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
        print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
        print(f"MoM (batch): err = {res['mom_err']:.4f},  mse = {res['mom_mse']:.4f}")
        print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
        print(f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
              f"best_mse = {res['exp_list_best_mse']:.4f}, "
              f"#cands = {res['exp_list_num_cands']}")

    return all_results

if __name__ == "__main__":
    sweep_alpha_uniform()
    # check_ols_recovery()  # optional debug
