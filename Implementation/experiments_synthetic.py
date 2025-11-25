# experiments_synthetic.py
import os
import csv
import numpy as np
import networkx as nx   # for optional expander construction
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    # Ensure Tables/ exists
    tables_dir = "Tables"
    os.makedirs(tables_dir, exist_ok=True)

    # If csv_path has no directory component, put it under Tables/
    if not os.path.dirname(csv_path):
        csv_path = os.path.join(tables_dir, csv_path)

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
    exp_single_lambda: float = 1e-4,
    exp_list_T: int = 7,
    exp_list_R: int = 10,
    exp_list_lambda: float = 1e-3,
):
    """
    Run one synthetic experiment for a given alpha and outlier model.
    Returns a dict of metrics for all methods.

    use_networkx_expander:
        If True, Expander-1 and Expander-L will use a NetworkX-generated
        bipartite expander graph (same graph shared across seeds).

    n, d, outlier_scale:
        Global problem size and corruption strength.
    """
    # Expander-single parameters
    B_sketch = 1000
    exp_single_r = 8

    # Expander-list parameters (Algorithm 1)
    exp_list_r = 8
    exp_list_theta = 0.1
    exp_list_rho = 0.5
    exp_list_cluster_radius = 0.0  # no merging of candidate vectors

    # 1. Generate synthetic data
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    # 1.5 Construct NetworkX expander if requested
    if use_networkx_expander:
        G_expander = make_random_regular_bipartite_expander(
            n=n,
            B=B_sketch,
            dL=exp_single_dL,
            seed=123 + random_state,
        )
    else:
        G_expander = None

    # Test set for all methods
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
        "exp_single_dL": exp_single_dL,
        "exp_single_lambda": exp_single_lambda,
        "exp_list_T": exp_list_T,
        "exp_list_R": exp_list_R,
        "exp_list_lambda": exp_list_lambda,
    }

    # 2. Baselines: OLS
    ols = LinearRegression().fit(X, y)
    beta_ols = ols.coef_
    results["ols_err"] = np.linalg.norm(beta_ols - w_star)
    results["ols_mse"] = mean_squared_error(y_test, X_test @ beta_ols)

    # 3. Sklearn baselines
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

    # 4. Expander-single (1 seed)
    beta_exp_single = expander_sketch_regression_single_seed(
        X,
        y,
        alpha=alpha,
        B=B_sketch,
        r=exp_single_r,
        dL=exp_single_dL,
        lambda_reg=exp_single_lambda,
        random_state=123,
        use_networkx=use_networkx_expander,
        graph=G_expander,
    )
    results["exp_single_err"] = np.linalg.norm(beta_exp_single - w_star)
    results["exp_single_mse"] = mean_squared_error(
        y_test,
        X_test @ beta_exp_single,
    )

    # 5. Expander-list (Algorithm 1)
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
        random_state=123,
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

def check_ols_recovery(alpha_no_outliers=1.0, alpha_many_outliers=0.3):
    X1, y1, w_star1, mask1, info1 = generate_regression_with_outliers(
        n=5000,
        d=20,
        alpha=alpha_no_outliers,
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=123,
    )
    ols1 = LinearRegression().fit(X1, y1)
    err1 = np.linalg.norm(ols1.coef_ - w_star1)
    print(f"[alpha={alpha_no_outliers}] OLS ||w_hat - w_star||_2 = {err1:.4f}")

    candidates_clean = expander_sketch_list_regression(
        X1,
        y1,
        alpha=alpha_no_outliers,
        r=5,
        B=None,
        dL=2,
        T=1,
        R=5,
        lambda_reg=1e-3,
        cluster_radius=0.0,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    print(
        f"[alpha={alpha_no_outliers}] Expander-list (clean) produced "
        f"{len(candidates_clean)} candidates"
    )
    for idx, beta in enumerate(candidates_clean):
        err = np.linalg.norm(beta - w_star1)
        print(f"  Clean cand {idx}: ||w_hat - w_star||_2 = {err:.4f}")

    X2, y2, w_star2, mask2, info2 = generate_regression_with_outliers(
        n=5000,
        d=20,
        alpha=alpha_many_outliers,
        outlier_mode="uniform",
        outlier_scale=10.0,
        random_state=123,
    )
    ols2 = LinearRegression().fit(X2, y2)
    err2 = np.linalg.norm(ols2.coef_ - w_star2)
    print(f"[alpha={alpha_many_outliers}] OLS ||w_hat - w_star||_2 = {err2:.4f}")

    rng = np.random.default_rng(123)
    X_test = rng.normal(size=(2000, X1.shape[1]))
    y_test1 = X_test @ w_star1
    y_test2 = X_test @ w_star2

    mse1 = mean_squared_error(y_test1, X_test @ ols1.coef_)
    mse2 = mean_squared_error(y_test2, X_test @ ols2.coef_)

    print(f"[alpha={alpha_no_outliers}] OLS test MSE = {mse1:.4f}")
    print(f"[alpha={alpha_many_outliers}] OLS test MSE = {mse2:.4f}")

    beta_ridge = fit_ridge(X2, y2, alpha=1.0)
    err_ridge = np.linalg.norm(beta_ridge - w_star2)
    mse_ridge = mean_squared_error(y_test2, X_test @ beta_ridge)
    print(f"[alpha={alpha_many_outliers}] Ridge ||w_hat - w_star||_2 = {err_ridge:.4f}")
    print(f"[alpha={alpha_many_outliers}] Ridge test MSE = {mse_ridge:.4f}")

    beta_huber = fit_huber(X2, y2, alpha=0.0001, epsilon=1.35)
    err_huber = np.linalg.norm(beta_huber - w_star2)
    mse_huber = mean_squared_error(y_test2, X_test @ beta_huber)
    print(f"[alpha={alpha_many_outliers}] Huber ||w_hat - w_star||_2 = {err_huber:.4f}")
    print(f"[alpha={alpha_many_outliers}] Huber test MSE = {mse_huber:.4f}")

    beta_ransac = fit_ransac(X2, y2, min_samples=None, residual_threshold=None, max_trials=100)
    err_ransac = np.linalg.norm(beta_ransac - w_star2)
    mse_ransac = mean_squared_error(y_test2, X_test @ beta_ransac)
    print(f"[alpha={alpha_many_outliers}] RANSAC ||w_hat - w_star||_2 = {err_ransac:.4f}")
    print(f"[alpha={alpha_many_outliers}] RANSAC test MSE = {mse_ransac:.4f}")

    beta_ts = fit_theilsen(X2, y2)
    err_ts = np.linalg.norm(beta_ts - w_star2)
    mse_ts = mean_squared_error(y_test2, X_test @ beta_ts)
    print(f"[alpha={alpha_many_outliers}] Theil-Sen ||w_hat - w_star||_2 = {err_ts:.4f}")
    print(f"[alpha={alpha_many_outliers}] Theil-Sen test MSE = {mse_ts:.4f}")

    beta_exp = expander_sketch_regression_single_seed(
        X2,
        y2,
        alpha=alpha_many_outliers,
        B=None,
        r=8,
        dL=3,
        lambda_reg=1e-4,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    err_exp = np.linalg.norm(beta_exp - w_star2)
    mse_exp = mean_squared_error(y_test2, X_test @ beta_exp)

    print(f"[alpha={alpha_many_outliers}] Expander-single ||w_hat - w_star||_2 = {err_exp:.4f}")
    print(f"[alpha={alpha_many_outliers}] Expander-single test MSE = {mse_exp:.4f}")

    candidates = expander_sketch_list_regression(
        X2,
        y2,
        alpha=alpha_many_outliers,
        r=8,
        B=None,
        dL=2,
        T=7,
        R=10,
        lambda_reg=1e-3,
        theta=0.1,
        rho=0.5,
        cluster_radius=0,
        random_state=123,
        use_networkx=False,
        graph=None,
    )

    print(f"[alpha={alpha_many_outliers}] Expander-list produced {len(candidates)} candidates")

    best_err = None
    best_mse = None
    for idx, beta in enumerate(candidates):
        err = np.linalg.norm(beta - w_star2)
        mse = mean_squared_error(y_test2, X_test @ beta)
        print(f"  Candidate {idx}: ||w_hat - w_star||_2 = {err:.4f}, test MSE = {mse:.4f}")
        if best_err is None or err < best_err:
            best_err = err
            best_mse = mse

    if best_err is not None:
        print(f"  Best candidate: param error = {best_err:.4f}, test MSE = {best_mse:.4f}")

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
    B_sketch = 1000,
):
    """
    Run a synthetic experiment for a given alpha and outlier model,
    but ONLY for Expander-L (no baselines, no Expander-1).

    This is used in hyperparameter sweeps for Expander-L.
    """
    
    # 1. Generate synthetic data
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    # 1.5 Construct NetworkX expander if requested
    if use_networkx_expander:
        G_expander = make_random_regular_bipartite_expander(
            n=n,
            B=B_sketch,
            dL=exp_list_dL,
            seed=123 + random_state,
        )
    else:
        G_expander = None

    # Test set for evaluation
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
        "exp_list_T": exp_list_T,
        "exp_list_R": exp_list_R,
        "exp_list_lambda": exp_list_lambda,
        "exp_list_dL": exp_list_dL,
        "exp_list_r": exp_list_r,
        "exp_list_theta": exp_list_theta,
        "exp_list_rho": exp_list_rho,
        "exp_list_cluster_radius": exp_list_cluster_radius,
    }

    # Expander-L (Algorithm 1)
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
        random_state=123,
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

def sweep_n_d(
    use_networkx_expander: bool = False,
    sweep_alpha: bool = True,
    csv_path: str = "Tables/results_sweep_n_d.csv",
):
    """
    Study scaling behavior by sweeping over (n, d),
    optionally also sweeping over alpha.

    Saves results to Tables/results_sweep_n_d.csv.
    """

    os.makedirs("Tables", exist_ok=True)

    n_values = [5000, 10000]
    d_values = [20, 50]

    # Fixed alpha for main paper OR sweep for appendix
    if sweep_alpha:
        alphas = [0.4, 0.3, 0.2, 0.1]
    else:
        alphas = [0.3]   # fixed α for cleaner scaling plots

    outlier_mode = "uniform"
    outlier_scale = 10.0

    all_results = []

    print("\n===== Sweep: scaling behavior over (n, d) =====")

    for n in n_values:
        for d in d_values:

            print("\n##################################################")
            print(f"(n, d) = ({n}, {d}), mode = {outlier_mode}")
            print("##################################################")

            for a in alphas:
                res = run_all_methods_for_alpha(
                    alpha=a,
                    outlier_mode=outlier_mode,
                    random_state=0,
                    use_networkx_expander=use_networkx_expander,
                    n=n,
                    d=d,
                    outlier_scale=outlier_scale,
                )

                all_results.append(res)
                append_result_wide(csv_path, res)

                print(f"\n--- n={n}, d={d}, alpha={a:.2f} ---")
                print(f"OLS        : mse = {res['ols_mse']:.4f}")
                print(f"Ridge      : mse = {res['ridge_mse']:.4f}")
                print(f"Huber      : mse = {res['huber_mse']:.4f}")
                print(f"RANSAC     : mse = {res['ransac_mse']:.4f}")
                print(f"Theil-Sen  : mse = {res['theilsen_mse']:.4f}")
                print(f"Expander-1 : mse = {res['exp_single_mse']:.4f}")
                print(f"Expander-L : mse = {res['exp_list_best_mse']:.4f}, "
                      f"#cands = {res['exp_list_num_cands']}")

    return all_results

def sweep_outlier_scale(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "results_outlier_scale.csv",
):
    """
    Sensitivity to outlier magnitude S (outlier_scale), for a fixed (n, d, alpha).

    We fix:
        - n = 5000
        - d = 20
        - outlier_mode = "uniform"
        - alpha (inlier fraction) = given argument (default 0.3)

    and sweep outlier_scale over a range of values.

    For each scale, we run all methods via run_all_methods_for_alpha
    and append a row to `csv_path` containing all metrics.
    """
    # You can tweak/extend this grid if you like
    scales = [5.0, 10.0, 20.0, 30.0]
    n = 5000
    d = 20
    outlier_mode = "uniform"

    all_results = []

    print("\n===== Sweep: sensitivity to outlier magnitude (fixed alpha) =====")
    print(f"alpha = {alpha}, mode = {outlier_mode}, n = {n}, d = {d}")
    print("scales =", scales)
    print("=================================================================")

    for scale in scales:
        print("\n##################################################")
        print(f"outlier_scale = {scale:.1f}, alpha = {alpha:.2f}")
        print("##################################################")

        res = run_all_methods_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )
        append_result_wide("results_sweep_outlier_scale.csv", res)
        all_results.append(res)

        # persist for plotting
        append_result_wide(csv_path, res)

        # compact console summary
        print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
        print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
        print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
        print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
        print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
        print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
        print(
            f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
            f"best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    return all_results

def sweep_alpha_uniform(use_networkx_expander: bool = False):
    """
    Run experiments for several alpha values under uniform outliers.
    """
    alphas = [0.4, 0.3, 0.2, 0.1]
    all_results = []

    for a in alphas:
        res = run_all_methods_for_alpha(
            alpha=a,
            outlier_mode="uniform",
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=5000,
            d=20,
            outlier_scale=10.0,
        )
        append_result_wide("results_uniform.csv", res)
        all_results.append(res)

        print("\n===========================================")
        print(
            f"Results for alpha = {a:.2f}, outlier_mode = uniform, "
            f"use_networkx={use_networkx_expander}"
        )
        print("-------------------------------------------")
        print(f"OLS        : err = {res['ols_err']:.4f},  mse = {res['ols_mse']:.4f}")
        print(f"Ridge      : err = {res['ridge_err']:.4f},  mse = {res['ridge_mse']:.4f}")
        print(f"Huber      : err = {res['huber_err']:.4f},  mse = {res['huber_mse']:.4f}")
        print(f"RANSAC     : err = {res['ransac_err']:.4f},  mse = {res['ransac_mse']:.4f}")
        print(f"Theil-Sen  : err = {res['theilsen_err']:.4f},  mse = {res['theilsen_mse']:.4f}")
        print(f"Expander-1 : err = {res['exp_single_err']:.4f},  mse = {res['exp_single_mse']:.4f}")
        print(
            f"Expander-L : best_err = {res['exp_list_best_err']:.4f}, "
            f"best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    return all_results

def sweep_R(use_networkx_expander: bool = False):
    """
    Sweep MSE vs R (number of seeds / candidates in Expander-L)
    on a fixed synthetic setting. Only Expander-L is evaluated.
    Also produces a PDF plot: Figures/mse_vs_R.pdf.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    R_values = list(range(1, 11))
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs R ===")
    for R in R_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_R=R,
        )
        append_result_wide("results_sweep_R.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(
            f"R = {R:2d} | "
            f"Expander-L best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    # Plot MSE vs R
    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(R_values, mse_vals, marker="o")
    plt.xticks(range(1, 10, 2))
    plt.xlabel("Number of seeds $R$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $R$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_R.pdf")
    plt.close()

    return all_results

def sweep_T(use_networkx_expander: bool = False):
    """
    Sweep MSE vs T (number of iterations in Expander-L)
    on a fixed synthetic setting. Only Expander-L is evaluated.
    Also produces a PDF plot: Figures/mse_vs_T.pdf.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    T_values = list(range(1, 16))
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs T ===")
    for T in T_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_T=T,
        )
        append_result_wide("results_sweep_T.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(
            f"T = {T:2d} | "
            f"Expander-L best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    # Plot MSE vs T
    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(T_values, mse_vals, marker="o")
    plt.xlabel("Number of filtering iterations $T$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $T$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_T.pdf")
    plt.close()

    return all_results

def sweep_dL(use_networkx_expander: bool = False):
    """
    Sweep MSE vs dL (left degree of the expander) for Expander-L only.
    Also produces a PDF plot: Figures/mse_vs_dL.pdf.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    dL_values = [2, 3, 4, 5, 6]
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs dL ===")
    for dL in dL_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_dL=dL,
        )
        append_result_wide("results_sweep_dL.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(
            f"dL = {dL:2d} | "
            f"Expander-L best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    # Plot MSE vs dL
    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(dL_values, mse_vals, marker="o")
    plt.xlabel("Expander left degree $d_L$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $d_L$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_dL.pdf")
    plt.close()

    return all_results

def sweep_lambda(use_networkx_expander: bool = False):
    """
    Sweep MSE vs lambda (ridge regularization) for Expander-L only.
    Also produces a PDF plot: Figures/mse_vs_lambda.pdf.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs lambda ===")
    for lam in lambdas:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_lambda=lam,
        )
        append_result_wide("results_sweep_lambda.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(
            f"lambda = {lam:.0e} | "
            f"Expander-L best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    # Plot MSE vs lambda (log-scale on x)
    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.semilogx(lambdas, mse_vals, marker="o")
    plt.xscale("log", base=10)   # <-- modern API
    plt.xlabel("Ridge parameter $\\lambda$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $\\lambda$")
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_lambda.pdf")
    plt.close()

    return all_results

def sweep_r(use_networkx_expander: bool = False):
    """
    Sweep MSE vs r (number of bucket repetitions per seed).
    Produces Figures/mse_vs_r.pdf.
    """
    alpha = 0.3
    n, d = 5000, 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    r_values = list(range(1, 16))
    mse_vals = []
    all_results = []

    print("\n=== Sweep: Expander-L MSE vs r ===")
    for r in r_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n, d=d,
            outlier_scale=outlier_scale,
            exp_list_r=r
        )
        append_result_wide("results_sweep_r.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(f"r = {r:2d} | MSE = {res['exp_list_best_mse']:.4f}")

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(r_values, mse_vals, marker="o")
    plt.xticks(range(1, 16, 2))
    plt.xlabel("Repetitions per seed $r$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $r$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_r_small.pdf")
    plt.close()

    return all_results

def sweep_theta(use_networkx_expander: bool = False):
    """
    Sweep MSE vs theta (pruning threshold).
    Produces Figures/mse_vs_theta.pdf.
    """
    alpha = 0.3
    n, d = 5000, 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    theta_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    mse_vals = []
    all_results = []

    print("\n=== Sweep: Expander-L MSE vs theta ===")
    for theta in theta_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n, d=d,
            outlier_scale=outlier_scale,
            exp_list_theta=theta
        )
        append_result_wide("results_sweep_theta.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(f"theta = {theta:.2f} | MSE = {res['exp_list_best_mse']:.4f}")

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(theta_values, mse_vals, marker="o")
    plt.xlabel("Pruning threshold $\\theta$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $\\theta$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_theta.pdf")
    plt.close()

    return all_results

def sweep_rho(use_networkx_expander: bool = False):
    """
    Sweep MSE vs rho (filter shrinkage factor).
    Produces Figures/mse_vs_rho.pdf.
    """
    alpha = 0.3
    n, d = 5000, 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    mse_vals = []
    all_results = []

    print("\n=== Sweep: Expander-L MSE vs rho ===")
    for rho in rho_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n, d=d,
            outlier_scale=outlier_scale,
            exp_list_rho=rho
        )
        append_result_wide("results_sweep_rho.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(f"rho = {rho:.1f} | MSE = {res['exp_list_best_mse']:.4f}")

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(rho_values, mse_vals, marker="o")
    plt.xlabel("Shrinkage factor $\\rho$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $\\rho$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_rho.pdf")
    plt.close()

    return all_results

def sweep_cluster_radius(use_networkx_expander: bool = False):
    """
    Sweep MSE vs cluster_radius (candidate merging).
    Produces Figures/mse_vs_cluster_radius.pdf.
    """
    alpha = 0.3
    n, d = 5000, 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    radius_values = [0, 1, 2, 3, 4]
    mse_vals = []
    all_results = []

    print("\n=== Sweep: Expander-L MSE vs cluster radius ===")
    for radius in radius_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n, d=d,
            outlier_scale=outlier_scale,
            exp_list_cluster_radius=radius
        )
        append_result_wide("results_sweep_cluster_radius.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(f"radius = {radius:.2f} | MSE = {res['exp_list_best_mse']:.4f}")

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(radius_values, mse_vals, marker="o")
    plt.xlabel("Cluster radius")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to cluster radius")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_cluster_radius.pdf")
    plt.close()

    return all_results

def sweep_B(use_networkx_expander: bool = False):
    """
    Sweep MSE vs B (number of buckets in the expander sketch) for Expander-L only.
    Produces a CSV file (results_sweep_B.csv) and a PDF plot: Figures/mse_vs_B.pdf.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    # Reasonable range around the theory-guided scale B ≍ (d/alpha) log(d/δ)
    B_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs B (number of buckets) ===")
    for B in B_values:
        res = run_expander_list_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            B_sketch=B,        # <--- vary B here
        )

        append_result_wide("results_sweep_B.csv", res)
        all_results.append(res)
        mse_vals.append(res["exp_list_best_mse"])

        print(
            f"B = {B:4d} | "
            f"Expander-L best_mse = {res['exp_list_best_mse']:.4f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

    # Plot MSE vs B
    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(B_values, mse_vals, marker="o")
    plt.xticks(B_values)
    plt.xlabel("Number of buckets $B$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $B$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_B.pdf")
    plt.close()

    return all_results

def plot_tsne_synthetic(
    n: int = 5000,
    d: int = 20,
    alpha: float = 0.3,
    outlier_mode: str = "uniform",
    outlier_scale: float = 10.0,
    random_state: int = 0,
    save_path: str = "tsne_synthetic_alpha0.3_uniform.png",
):
    """
    Generate a single synthetic dataset and visualize it with t-SNE.
    Defaults match the main experiment settings.
    """
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=random_state,
    )
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    inliers = inlier_mask
    outliers = ~inlier_mask

    plt.scatter(
        X_emb[inliers, 0],
        X_emb[inliers, 1],
        s=10,
        alpha=0.7,
        label="Inliers",
    )
    plt.scatter(
        X_emb[outliers, 0],
        X_emb[outliers, 1],
        s=10,
        alpha=0.7,
        label="Outliers",
    )

    plt.title(f"t-SNE of synthetic data (alpha = {alpha}, mode = {outlier_mode})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[t-SNE] Saved t-SNE plot to {save_path}")
    plt.close()

def sweep_outlier_scale_mse(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "Tables/results_outlier_scale.csv",
):
    """
    Sweep outlier magnitude S and:
      • Save full results (all methods) to Tables/
      • Produce a plot of Expander-L test MSE vs S (only Expander-L)
    """

    os.makedirs("Tables", exist_ok=True)
    scales = [1, 5, 10, 15, 20, 25, 30]

    n = 5000
    d = 20
    outlier_mode = "uniform"

    all_results = []
    mse_expL = []

    print("\n===== Sweep: sensitivity to outlier magnitude (fixed alpha) =====")
    print(f"alpha = {alpha}, mode = {outlier_mode}, n = {n}, d = {d}")
    print("scales =", scales)
    print("=================================================================")

    for scale in scales:

        print("\n##################################################")
        print(f"outlier_scale = {scale:.1f}, alpha = {alpha:.2f}")
        print("##################################################")

        res = run_all_methods_for_alpha(
            alpha=alpha,
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )

        all_results.append(res)
        
        # Save Expander-L MSE for plot
        mse_expL.append(res["exp_list_best_mse"])

        # Console summary
        print(f"Expander-L : best_mse = {res['exp_list_best_mse']:.4f}, "
              f"#cands = {res['exp_list_num_cands']}")

    plt.figure(figsize=(7, 5))
    plt.plot(scales, mse_expL, marker="o", linewidth=2, label="Expander-L")

    plt.xlabel("Outlier magnitude $S$")
    plt.ylabel("Test MSE")
    plt.title(f"Expander-L: sensitivity to outlier magnitude (alpha={alpha})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_fig_path = "Figures/mse_vs_outlier_scale.pdf"
    plt.savefig(save_fig_path, dpi=300)
    plt.close()

    print(f"[Plot] Saved Expander-L MSE-vs-outlier-scale curve to {save_fig_path}")

    return all_results

def plot_projection_vs_y(
    n: int = 5000,
    d: int = 20,
    alpha: float = 0.3,
    outlier_mode: str = "uniform",
    outlier_scale: float = 10.0,
    random_state: int = 0,
    save_path: str = "projection_vs_y.pdf",
):
    """
    Plot (x_i^T w*, y_i) for inliers and outliers.
    Inliers lie along a clean linear trend.
    Outliers appear scattered.

    This visualization communicates the structure
    of the synthetic list-decodable regression dataset.
    """

    # Ensure Figures/ exists
    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)

    # If save_path has no directory component, place it in Figures/
    if not os.path.dirname(save_path):
        save_path = os.path.join(figures_dir, save_path)

    # Generate synthetic dataset
    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    # Compute projection z_i = x_i^T w*
    z = X @ w_star

    # Split inliers/outliers
    inliers = inlier_mask
    outliers = ~inlier_mask

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(z[inliers], y[inliers], s=10, alpha=0.7, label="Inliers")
    plt.scatter(z[outliers], y[outliers], s=10, alpha=0.7, label="Outliers")

    plt.xlabel(r"$x_i^\top w^\star$")
    plt.ylabel(r"$y_i$")
    plt.title(f"Projection vs. Response (alpha={alpha}, mode={outlier_mode})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"[Plot] Saved projection vs y plot to {save_path}")
    plt.close()

def sweep_alpha_mse(
    use_networkx_expander: bool = False,
    outlier_mode: str = "uniform",
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    save_path: str = "mse_vs_alpha_uniform.pdf",
):
    """
    Sweep alpha from 1.0 down to 0.10 in steps of 0.05 and
    plot *only* Expander-L's test MSE vs alpha.

    This is the clean performance curve for the main algorithm.
    """
    alphas = np.arange(1.0, 0.0, -0.05)
    alphas = np.round(alphas, 2)

    mse_expL = []

    print("\n#########################")
    print("Sweep: Expander-L MSE vs alpha")
    print("#########################")

    for a in alphas:
        res = run_expander_list_for_alpha(
            alpha=float(a),
            outlier_mode=outlier_mode,
            random_state=0,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
        )

        print(
            f"alpha = {a:.2f} | "
            f"Exp-L best_mse = {res['exp_list_best_mse']:.3f}, "
            f"#cands = {res['exp_list_num_cands']}"
        )

        mse_expL.append(res["exp_list_best_mse"])

    # Ensure Figures/ exists
    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)

    # If save_path has no directory, place it in Figures/
    if not os.path.dirname(save_path):
        save_path = os.path.join(figures_dir, save_path)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(alphas, mse_expL, marker="o", label="Expander-L (best)")

    plt.gca().invert_xaxis()
    plt.xlabel(r"Inlier fraction $\alpha$")
    plt.ylabel("Test MSE")
    plt.title(f"Expander-L performance vs α (mode={outlier_mode}, n={n}, d={d})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Plot] Saved Expander-L MSE-vs-alpha curve to {save_path}")
    plt.close()


if __name__ == "__main__":
    # sweep_alpha_uniform(use_networkx_expander=False)
    # sweep_n_d(use_networkx_expander=False)
    # sweep_outlier_scale(use_networkx_expander=False)
    # sweep_outlier_scale_mse(use_networkx_expander= False)
    sweep_R(use_networkx_expander=False)
    # sweep_T(use_networkx_expander=False)
    # sweep_dL(use_networkx_expander=False)
    # sweep_lambda(use_networkx_expander=False)
    sweep_r(use_networkx_expander= False)
    sweep_theta(use_networkx_expander= False)
    # sweep_rho(use_networkx_expander= False)
    sweep_B(use_networkx_expander=False)
    # sweep_cluster_radius(use_networkx_expander= False)
    # plot_tsne_synthetic()
    # sweep_alpha_mse()
    # plot_projection_vs_y()
    
    # Optional detailed check:
    # check_ols_recovery()
