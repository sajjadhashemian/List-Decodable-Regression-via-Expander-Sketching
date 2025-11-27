"""
Sweep utilities for synthetic experiments.
Each sweep runs multi-seed variants and writes per-seed + aggregate rows.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from synthetic_core import (
    append_result_wide,
    run_all_methods_for_alpha_multi_seed,
    run_expander_list_for_alpha_multi_seed,
)

def sweep_n_d(
    use_networkx_expander: bool = False,
    sweep_alpha: bool = True,
    csv_path: str = "Tables/results_sweep_n_d.csv",
    seeds = (0, 1, 2, 3, 4),
):
    """
    Study scaling behavior by sweeping over (n, d),
    optionally also sweeping over alpha, using ONLY Expander-L
    (averaged over seeds).
    """
    os.makedirs("Tables", exist_ok=True)

    n_values = [5000, 6500]
    d_values = [20, 50]

    alphas = [0.4, 0.3, 0.2, 0.1] if sweep_alpha else [0.3]
    outlier_mode = "uniform"
    outlier_scale = 10.0

    all_results = []

    print("\n===== Sweep: scaling behavior over (n, d) [Expander-L only] =====")
    for n in n_values:
        for d in d_values:
            print("\n##################################################")
            print(f"(n, d) = ({n}, {d}), mode = {outlier_mode}")
            print("##################################################")

            for a in alphas:
                agg, per_seed = run_expander_list_for_alpha_multi_seed(
                    alpha=a,
                    outlier_mode=outlier_mode,
                    seeds=seeds,
                    use_networkx_expander=use_networkx_expander,
                    n=n,
                    d=d,
                    outlier_scale=outlier_scale,
                )

                all_results.append(agg)
                for res in per_seed:
                    append_result_wide(csv_path, res)
                append_result_wide(csv_path, agg)

                print(f"\n--- n={n}, d={d}, alpha={a:.2f} ---")
                print(
                    f"Expander-L : mse = {agg['exp_list_best_mse_mean']:.4f} "
                    f"(± {agg['exp_list_best_mse_std']:.4f}), "
                    f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
                )

    return all_results

def sweep_outlier_scale(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "results_outlier_scale.csv",
    seeds = (0, 1, 2, 3, 4),
):
    """
    Sensitivity to outlier magnitude S (outlier_scale), for a fixed (n, d, alpha).
    Runs all methods and logs per-seed and aggregate results.
    """
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

        agg, per_seed = run_all_methods_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )
        for res in per_seed:
            append_result_wide("results_sweep_outlier_scale.csv", res)
            append_result_wide(csv_path, res)
        append_result_wide("results_sweep_outlier_scale.csv", agg)
        append_result_wide(csv_path, agg)
        all_results.append(agg)

        print(
            f"OLS        : mse = {agg['ols_mse_mean']:.4f} "
            f"(± {agg['ols_mse_std']:.4f})"
        )
        print(
            f"Ridge      : mse = {agg['ridge_mse_mean']:.4f} "
            f"(± {agg['ridge_mse_std']:.4f})"
        )
        print(
            f"Huber      : mse = {agg['huber_mse_mean']:.4f} "
            f"(± {agg['huber_mse_std']:.4f})"
        )
        print(
            f"RANSAC     : mse = {agg['ransac_mse_mean']:.4f} "
            f"(± {agg['ransac_mse_std']:.4f})"
        )
        print(
            f"Theil-Sen  : mse = {agg['theilsen_mse_mean']:.4f} "
            f"(± {agg['theilsen_mse_std']:.4f})"
        )
        print(
            f"Expander-1 : mse = {agg['exp_single_mse_mean']:.4f} "
            f"(± {agg['exp_single_mse_std']:.4f})"
        )
        print(
            f"Expander-L : best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    return all_results

def sweep_alpha_uniform(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Run experiments for several alpha values under uniform outliers.
    """
    alphas = [0.4, 0.3, 0.2, 0.1]
    all_results = []

    for a in alphas:
        agg, per_seed = run_all_methods_for_alpha_multi_seed(
            alpha=a,
            outlier_mode="uniform",
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=5000,
            d=20,
            outlier_scale=10.0,
        )
        for res in per_seed:
            append_result_wide("results_uniform.csv", res)
        append_result_wide("results_uniform.csv", agg)
        all_results.append(agg)

        print("\n===========================================")
        print(
            f"Results for alpha = {a:.2f}, outlier_mode = uniform, "
            f"use_networkx={use_networkx_expander}"
        )
        print("-------------------------------------------")
        print(
            f"OLS        : mse = {agg['ols_mse_mean']:.4f} "
            f"(± {agg['ols_mse_std']:.4f})"
        )
        print(
            f"Ridge      : mse = {agg['ridge_mse_mean']:.4f} "
            f"(± {agg['ridge_mse_std']:.4f})"
        )
        print(
            f"Huber      : mse = {agg['huber_mse_mean']:.4f} "
            f"(± {agg['huber_mse_std']:.4f})"
        )
        print(
            f"RANSAC     : mse = {agg['ransac_mse_mean']:.4f} "
            f"(± {agg['ransac_mse_std']:.4f})"
        )
        print(
            f"Theil-Sen  : mse = {agg['theilsen_mse_mean']:.4f} "
            f"(± {agg['theilsen_mse_std']:.4f})"
        )
        print(
            f"Expander-1 : mse = {agg['exp_single_mse_mean']:.4f} "
            f"(± {agg['exp_single_mse_std']:.4f})"
        )
        print(
            f"Expander-L : best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    return all_results

def sweep_R(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs R (number of seeds / candidates in Expander-L)
    on a fixed synthetic setting. Only Expander-L is evaluated,
    averaged over the given seeds.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    R_values = list(range(1, 16))
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs R ===")
    for R in R_values:
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_R=R,
        )
        for res in per_seed:
            append_result_wide("results_sweep_R.csv", res)
        append_result_wide("results_sweep_R.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"R = {R:2d} | "
            f"Expander-L best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(R_values, mse_vals, marker="o")
    plt.xticks(range(1, 16, 2))
    plt.xlabel("Number of seeds $R$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $R$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_R.pdf")
    plt.close()

    return all_results

def sweep_T(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs T (number of iterations in Expander-L),
    averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_T=T,
        )
        for res in per_seed:
            append_result_wide("results_sweep_T.csv", res)
        append_result_wide("results_sweep_T.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"T = {T:2d} | "
            f"Expander-L best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

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

def sweep_dL(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs dL (left degree of the expander) for Expander-L only,
    averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_dL=dL,
        )
        for res in per_seed:
            append_result_wide("results_sweep_dL.csv", res)
        append_result_wide("results_sweep_dL.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"dL = {dL:2d} | "
            f"Expander-L best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

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

def sweep_lambda(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs lambda (ridge regularization) for Expander-L only,
    averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_lambda=lam,
        )
        for res in per_seed:
            append_result_wide("results_sweep_lambda.csv", res)
        append_result_wide("results_sweep_lambda.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"lambda = {lam:.0e} | "
            f"Expander-L best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.semilogx(lambdas, mse_vals, marker="o")
    plt.xscale("log", base=10)
    plt.xlabel("Ridge parameter $\\lambda$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $\\lambda$")
    plt.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_lambda.pdf")
    plt.close()

    return all_results

def sweep_r(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs r (number of bucket repetitions per seed),
    averaged over seeds.
    """
    alpha = 0.3
    n, d = 5000, 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    r_values = list(range(1, 11))
    mse_vals = []
    all_results = []

    print("\n=== Sweep: Expander-L MSE vs r ===")
    for r in r_values:
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_r=r,
        )
        for res in per_seed:
            append_result_wide("results_sweep_r.csv", res)
        append_result_wide("results_sweep_r.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"r = {r:2d} | "
            f"MSE = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f})"
        )

    os.makedirs("Figures", exist_ok=True)
    plt.figure()
    plt.plot(r_values, mse_vals, marker="o")
    plt.xticks(range(1, 11, 2))
    plt.xlabel("Repetitions per seed $r$")
    plt.ylabel("Test MSE")
    plt.title("Expander-L: sensitivity to $r$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Figures/mse_vs_r_small.pdf")
    plt.close()

    return all_results

def sweep_theta(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs theta (pruning threshold), averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_theta=theta,
        )
        for res in per_seed:
            append_result_wide("results_sweep_theta.csv", res)
        append_result_wide("results_sweep_theta.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"theta = {theta:.2f} | "
            f"MSE = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f})"
        )

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

def sweep_rho(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs rho (filter shrinkage factor), averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_rho=rho,
        )
        for res in per_seed:
            append_result_wide("results_sweep_rho.csv", res)
        append_result_wide("results_sweep_rho.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"rho = {rho:.1f} | "
            f"MSE = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f})"
        )

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

def sweep_cluster_radius(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs cluster_radius (candidate merging), averaged over seeds.
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
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            exp_list_cluster_radius=radius,
        )
        for res in per_seed:
            append_result_wide("results_sweep_cluster_radius.csv", res)
        append_result_wide("results_sweep_cluster_radius.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"radius = {radius:.2f} | "
            f"MSE = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f})"
        )

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

def sweep_B(use_networkx_expander: bool = False, seeds = (0, 1, 2, 3, 4)):
    """
    Sweep MSE vs B (number of buckets in the expander sketch) for Expander-L,
    averaged over seeds.
    """
    alpha = 0.3
    n = 5000
    d = 20
    outlier_mode = "uniform"
    outlier_scale = 10.0

    B_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    all_results = []
    mse_vals = []

    print("\n=== Sweep: Expander-L MSE vs B (number of buckets) ===")
    for B in B_values:
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
            B_sketch=B,
        )

        for res in per_seed:
            append_result_wide("results_sweep_B.csv", res)
        append_result_wide("results_sweep_B.csv", agg)

        all_results.append(agg)
        mse_vals.append(agg["exp_list_best_mse_mean"])

        print(
            f"B = {B:4d} | "
            f"Expander-L best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']}"
        )

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

def sweep_alpha_mse(
    use_networkx_expander: bool = False,
    outlier_mode: str = "uniform",
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    save_path: str = "mse_vs_alpha_uniform.pdf",
    seeds = (0, 1, 2, 3, 4),
):
    """
    Sweep alpha from 1.0 down to 0.10 in steps of 0.05 and
    plot *only* Expander-L's test MSE vs alpha, averaged over seeds.
    """
    alphas = np.arange(1.0, 0.0, -0.05)
    alphas = np.round(alphas, 2)

    mse_expL = []

    print("\n#########################")
    print("Sweep: Expander-L MSE vs alpha")
    print("#########################")

    for a in alphas:
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=float(a),
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
        )

        print(
            f"alpha = {a:.2f} | "
            f"Exp-L best_mse = {agg['exp_list_best_mse_mean']:.3f} "
            f"(± {agg['exp_list_best_mse_std']:.3f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

        mse_expL.append(agg["exp_list_best_mse_mean"])

    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)
    if not os.path.dirname(save_path):
        save_path = os.path.join(figures_dir, save_path)

    plt.figure(figsize=(7, 5))
    plt.plot(alphas, mse_expL, marker="o", label="Expander-L (best, mean over seeds)")

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

    return mse_expL

def sweep_outlier_scale_mse(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "Tables/results_outlier_scale.csv",
    seeds = (0, 1, 2, 3, 4),
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

        agg, per_seed = run_all_methods_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )

        all_results.append(agg)
        mse_expL.append(agg["exp_list_best_mse_mean"])

        for res in per_seed:
            append_result_wide(csv_path, res)
        append_result_wide(csv_path, agg)

        print(
            f"Expander-L : best_mse = {agg['exp_list_best_mse_mean']:.4f} "
            f"(± {agg['exp_list_best_mse_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

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

def sweep_alpha_uniform_param_err(
    use_networkx_expander: bool = False,
    seeds=(0, 1, 2, 3, 4),
    csv_path: str = "Tables/results_alpha_paramerr_uniform.csv",
):
    """
    Scenario A, Part 1:
      ‣ Parameter error ||w_hat - w*||_2 vs inlier fraction alpha,
        for all methods, averaged over seeds.

    Saves:
      • CSV: Tables/results_alpha_paramerr_uniform.csv
      • Figure: Figures/paramerr_vs_alpha_all_methods.pdf
    """
    os.makedirs("Tables", exist_ok=True)
    os.makedirs("Figures", exist_ok=True)

    alphas = [0.4, 0.3, 0.2, 0.1]
    all_results = []

    # For plotting
    alpha_vals = []
    ols_err_means = []
    ridge_err_means = []
    huber_err_means = []
    ransac_err_means = []
    theilsen_err_means = []
    exp1_err_means = []
    expl_err_means = []

    print("\n===== Sweep (param error): all methods vs alpha =====")
    for a in alphas:
        agg, per_seed = run_all_methods_for_alpha_multi_seed(
            alpha=a,
            outlier_mode="uniform",
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=5000,
            d=20,
            outlier_scale=10.0,
        )

        # Save per-seed + aggregated to CSV
        for res in per_seed:
            append_result_wide(csv_path, res)
        append_result_wide(csv_path, agg)
        all_results.append(agg)

        alpha_vals.append(a)
        ols_err_means.append(agg["ols_err_mean"])
        ridge_err_means.append(agg["ridge_err_mean"])
        huber_err_means.append(agg["huber_err_mean"])
        ransac_err_means.append(agg["ransac_err_mean"])
        theilsen_err_means.append(agg["theilsen_err_mean"])
        exp1_err_means.append(agg["exp_single_err_mean"])
        expl_err_means.append(agg["exp_list_best_err_mean"])

        print("\n-------------------------------------------")
        print(
            f"alpha = {a:.2f}, mode = uniform, use_networkx={use_networkx_expander}"
        )
        print(
            f"OLS        : ||ŵ - w*|| = {agg['ols_err_mean']:.4f} "
            f"(± {agg['ols_err_std']:.4f})"
        )
        print(
            f"Ridge      : ||ŵ - w*|| = {agg['ridge_err_mean']:.4f} "
            f"(± {agg['ridge_err_std']:.4f})"
        )
        print(
            f"Huber      : ||ŵ - w*|| = {agg['huber_err_mean']:.4f} "
            f"(± {agg['huber_err_std']:.4f})"
        )
        print(
            f"RANSAC     : ||ŵ - w*|| = {agg['ransac_err_mean']:.4f} "
            f"(± {agg['ransac_err_std']:.4f})"
        )
        print(
            f"Theil-Sen  : ||ŵ - w*|| = {agg['theilsen_err_mean']:.4f} "
            f"(± {agg['theilsen_err_std']:.4f})"
        )
        print(
            f"Expander-1 : ||ŵ - w*|| = {agg['exp_single_err_mean']:.4f} "
            f"(± {agg['exp_single_err_std']:.4f})"
        )
        print(
            f"Expander-L : ||ŵ - w*|| = {agg['exp_list_best_err_mean']:.4f} "
            f"(± {agg['exp_list_best_err_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    # Plot: parameter error vs alpha (all methods)
    plt.figure(figsize=(7, 5))
    plt.plot(alpha_vals, ols_err_means, marker="o", label="OLS")
    plt.plot(alpha_vals, ridge_err_means, marker="o", label="Ridge")
    plt.plot(alpha_vals, huber_err_means, marker="o", label="Huber")
    plt.plot(alpha_vals, ransac_err_means, marker="o", label="RANSAC")
    plt.plot(alpha_vals, theilsen_err_means, marker="o", label="Theil-Sen")
    plt.plot(alpha_vals, exp1_err_means, marker="o", label="Expander-1")
    plt.plot(alpha_vals, expl_err_means, marker="o", label="Expander-L (best)")

    plt.gca().invert_xaxis()
    plt.xlabel(r"Inlier fraction $\alpha$")
    plt.ylabel(r"Parameter error $\|\hat w - w^\star\|_2$")
    plt.title("Parameter error vs inlier fraction (all methods)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = "Figures/paramerr_vs_alpha_all_methods.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved parameter-error vs alpha (all methods) to {fig_path}")

    return all_results

def sweep_alpha_param_err_expanderL(
    use_networkx_expander: bool = False,
    outlier_mode: str = "uniform",
    n: int = 5000,
    d: int = 20,
    outlier_scale: float = 10.0,
    seeds=(0, 1, 2, 3, 4),
    csv_path: str = "Tables/results_alpha_paramerr_expanderL.csv",
    save_path: str = "paramerr_vs_alpha_expanderL.pdf",
):
    """
    Scenario A, Part 2:
      ‣ Fine-grained parameter error of Expander-L vs alpha, averaged over seeds.

    Sweeps alpha on a fine grid (1.0 down to 0.10, step 0.05).

    Saves:
      • CSV: Tables/results_alpha_paramerr_expanderL.csv
      • Figure: Figures/paramerr_vs_alpha_expanderL.pdf
    """
    os.makedirs("Tables", exist_ok=True)
    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)

    alphas = np.arange(1.0, 0.0, -0.05)
    alphas = np.round(alphas, 2)

    alpha_vals = []
    expl_err_means = []
    expl_err_stds = []

    print("\n#########################")
    print("Sweep (param error): Expander-L vs alpha (fine grid)")
    print("#########################")

    for a in alphas:
        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=float(a),
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=outlier_scale,
        )

        for res in per_seed:
            append_result_wide(csv_path, res)
        append_result_wide(csv_path, agg)

        alpha_vals.append(float(a))
        expl_err_means.append(agg["exp_list_best_err_mean"])
        expl_err_stds.append(agg["exp_list_best_err_std"])

        print(
            f"alpha = {a:.2f} | "
            f"Exp-L ||ŵ - w*|| = {agg['exp_list_best_err_mean']:.3f} "
            f"(± {agg['exp_list_best_err_std']:.3f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    if not os.path.dirname(save_path):
        save_path = os.path.join(figures_dir, save_path)

    plt.figure(figsize=(7, 5))
    plt.plot(alpha_vals, expl_err_means, marker="o", label="Expander-L (best, mean over seeds)")

    plt.gca().invert_xaxis()
    plt.xlabel(r"Inlier fraction $\alpha$")
    plt.ylabel(r"Parameter error $\|\hat w - w^\star\|_2$")
    plt.title(f"Expander-L parameter error vs α (mode={outlier_mode}, n={n}, d={d})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Plot] Saved Expander-L parameter-error vs alpha curve to {save_path}")
    plt.close()

def sweep_outlier_scale_param_err(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "Tables/results_paramerr_outlier_scale_all.csv",
    seeds=(0, 1, 2, 3, 4),
):
    """
    Scenario B, Part 1:
      ‣ Parameter error ||w_hat - w*||_2 vs outlier magnitude S (outlier_scale),
        for all methods, averaged over seeds, at fixed alpha.

    Saves:
      • CSV: Tables/results_paramerr_outlier_scale_all.csv
      • Figure: Figures/paramerr_vs_outlier_scale_all_methods.pdf
    """
    os.makedirs("Tables", exist_ok=True)
    os.makedirs("Figures", exist_ok=True)

    scales = [5.0, 10.0, 20.0, 30.0]
    n = 5000
    d = 20
    outlier_mode = "uniform"

    all_results = []

    S_vals = []
    ols_err_means = []
    ridge_err_means = []
    huber_err_means = []
    ransac_err_means = []
    theilsen_err_means = []
    exp1_err_means = []
    expl_err_means = []

    print("\n===== Sweep (param error): all methods vs outlier magnitude S =====")
    print(f"alpha = {alpha}, mode = {outlier_mode}, n = {n}, d = {d}")
    print("S values =", scales)
    print("=================================================================")

    for scale in scales:
        print("\n##################################################")
        print(f"outlier_scale = {scale:.1f}, alpha = {alpha:.2f}")
        print("##################################################")

        agg, per_seed = run_all_methods_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )

        all_results.append(agg)
        S_vals.append(scale)

        # Save rows for tables
        for res in per_seed:
            append_result_wide(csv_path, res)
        append_result_wide(csv_path, agg)

        ols_err_means.append(agg["ols_err_mean"])
        ridge_err_means.append(agg["ridge_err_mean"])
        huber_err_means.append(agg["huber_err_mean"])
        ransac_err_means.append(agg["ransac_err_mean"])
        theilsen_err_means.append(agg["theilsen_err_mean"])
        exp1_err_means.append(agg["exp_single_err_mean"])
        expl_err_means.append(agg["exp_list_best_err_mean"])

        print(
            f"OLS        : ||ŵ - w*|| = {agg['ols_err_mean']:.4f} "
            f"(± {agg['ols_err_std']:.4f})"
        )
        print(
            f"Ridge      : ||ŵ - w*|| = {agg['ridge_err_mean']:.4f} "
            f"(± {agg['ridge_err_std']:.4f})"
        )
        print(
            f"Huber      : ||ŵ - w*|| = {agg['huber_err_mean']:.4f} "
            f"(± {agg['huber_err_std']:.4f})"
        )
        print(
            f"RANSAC     : ||ŵ - w*|| = {agg['ransac_err_mean']:.4f} "
            f"(± {agg['ransac_err_std']:.4f})"
        )
        print(
            f"Theil-Sen  : ||ŵ - w*|| = {agg['theilsen_err_mean']:.4f} "
            f"(± {agg['theilsen_err_std']:.4f})"
        )
        print(
            f"Expander-1 : ||ŵ - w*|| = {agg['exp_single_err_mean']:.4f} "
            f"(± {agg['exp_single_err_std']:.4f})"
        )
        print(
            f"Expander-L : ||ŵ - w*|| = {agg['exp_list_best_err_mean']:.4f} "
            f"(± {agg['exp_list_best_err_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    # Plot: parameter error vs S for all methods
    plt.figure(figsize=(7, 5))
    plt.plot(S_vals, ols_err_means, marker="o", label="OLS")
    plt.plot(S_vals, ridge_err_means, marker="o", label="Ridge")
    plt.plot(S_vals, huber_err_means, marker="o", label="Huber")
    plt.plot(S_vals, ransac_err_means, marker="o", label="RANSAC")
    plt.plot(S_vals, theilsen_err_means, marker="o", label="Theil-Sen")
    plt.plot(S_vals, exp1_err_means, marker="o", label="Expander-1")
    plt.plot(S_vals, expl_err_means, marker="o", label="Expander-L (best)")

    plt.xlabel("Outlier magnitude $S$")
    plt.ylabel(r"Parameter error $\|\hat w - w^\star\|_2$")
    plt.title(f"Parameter error vs outlier magnitude (alpha={alpha})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = "Figures/paramerr_vs_outlier_scale_all_methods.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved parameter-error vs outlier-scale (all methods) to {fig_path}")

    return all_results

def sweep_outlier_scale_param_err_expanderL(
    use_networkx_expander: bool = False,
    alpha: float = 0.3,
    csv_path: str = "Tables/results_paramerr_outlier_scale_expanderL.csv",
    seeds=(0, 1, 2, 3, 4),
    save_fig_path: str = "paramerr_vs_outlier_scale_expanderL.pdf",
):
    """
    Scenario B, Part 2:
      ‣ Fine-grained parameter error of Expander-L vs outlier magnitude S,
        averaged over seeds, at fixed alpha.

    Uses a finer grid of S values.

    Saves:
      • CSV: Tables/results_paramerr_outlier_scale_expanderL.csv
      • Figure: Figures/paramerr_vs_outlier_scale_expanderL.pdf
    """
    os.makedirs("Tables", exist_ok=True)
    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)

    scales = [1, 5, 10, 15, 20, 25, 30]
    n = 5000
    d = 20
    outlier_mode = "uniform"

    all_results = []
    S_vals = []
    expl_err_means = []
    expl_err_stds = []

    print("\n===== Sweep (param error): Expander-L vs outlier magnitude S =====")
    print(f"alpha = {alpha}, mode = {outlier_mode}, n = {n}, d = {d}")
    print("S values =", scales)
    print("=================================================================")

    for scale in scales:
        print("\n##################################################")
        print(f"outlier_scale = {scale:.1f}, alpha = {alpha:.2f}")
        print("##################################################")

        agg, per_seed = run_expander_list_for_alpha_multi_seed(
            alpha=alpha,
            outlier_mode=outlier_mode,
            seeds=seeds,
            use_networkx_expander=use_networkx_expander,
            n=n,
            d=d,
            outlier_scale=scale,
        )

        all_results.append(agg)
        S_vals.append(scale)
        expl_err_means.append(agg["exp_list_best_err_mean"])
        expl_err_stds.append(agg["exp_list_best_err_std"])

        for res in per_seed:
            append_result_wide(csv_path, res)
        append_result_wide(csv_path, agg)

        print(
            f"Expander-L : ||ŵ - w*|| = {agg['exp_list_best_err_mean']:.4f} "
            f"(± {agg['exp_list_best_err_std']:.4f}), "
            f"#cands = {agg['exp_list_num_cands_mean']:.1f}"
        )

    if not os.path.dirname(save_fig_path):
        save_fig_path = os.path.join(figures_dir, save_fig_path)

    plt.figure(figsize=(7, 5))
    plt.plot(S_vals, expl_err_means, marker="o", linewidth=2, label="Expander-L")

    plt.xlabel("Outlier magnitude $S$")
    plt.ylabel(r"Parameter error $\|\hat w - w^\star\|_2$")
    plt.title(f"Expander-L: parameter error vs outlier magnitude (alpha={alpha})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.close()

    print(f"[Plot] Saved Expander-L param-error vs outlier-scale curve to {save_fig_path}")

    return all_results
