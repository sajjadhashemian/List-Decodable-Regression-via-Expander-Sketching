"""
Thin entrypoint that delegates to modularized code:
  - synthetic_core.py   (single- and multi-seed experiment runners)
  - synthetic_sweeps.py (all sweep_* utilities)
  - synthetic_plots.py  (plotting helpers)

The previous monolithic implementation is preserved in
experiments_synthetic_legacy.py for easy rollback.
"""

from synthetic_sweeps import (
    sweep_n_d,
    sweep_outlier_scale,
    sweep_alpha_uniform,
    sweep_outlier_scale_mse,
    sweep_R,
    sweep_T,
    sweep_dL,
    sweep_lambda,
    sweep_r,
    sweep_theta,
    sweep_rho,
    sweep_cluster_radius,
    sweep_B,
    sweep_alpha_mse,
    sweep_alpha_uniform_param_err,
    sweep_alpha_param_err_expanderL,
    sweep_outlier_scale_param_err,
    sweep_outlier_scale_param_err_expanderL,
)

from synthetic_plots import plot_tsne_synthetic, plot_projection_vs_y


def main():
    seeds = [0, 1, 2, 3, 4]

    # Main sweeps (toggle as needed)
    # sweep_alpha_uniform(use_networkx_expander=False, seeds=seeds)
    sweep_n_d(use_networkx_expander=False, seeds=seeds)
    # sweep_outlier_scale(use_networkx_expander=False, seeds=seeds)
    # sweep_outlier_scale_mse(use_networkx_expander=False, seeds=seeds)
    #sweep_R(use_networkx_expander=False, seeds=seeds)

    # Optional sweeps (uncomment to run)
    # sweep_T(use_networkx_expander=False, seeds=seeds)
    # sweep_dL(use_networkx_expander=False, seeds=seeds)
    # sweep_lambda(use_networkx_expander=False, seeds=seeds)
    # sweep_r(use_networkx_expander=False, seeds=seeds)
    # sweep_theta(use_networkx_expander=False, seeds=seeds)
    # sweep_rho(use_networkx_expander=False, seeds=seeds)
    # sweep_B(use_networkx_expander=False, seeds=seeds)
    # sweep_cluster_radius(use_networkx_expander=False, seeds=seeds)
    # sweep_alpha_mse(use_networkx_expander=False, seeds=seeds)
    # sweep_alpha_uniform_param_err(use_networkx_expander=False, seeds=seeds)
    # sweep_alpha_param_err_expanderL(use_networkx_expander=False, seeds=seeds)
    # sweep_outlier_scale_param_err(use_networkx_expander=False, seeds=seeds)
    # sweep_outlier_scale_param_err_expanderL(use_networkx_expander=False, seeds=seeds)
    # Optional plots
    # plot_tsne_synthetic()
    # plot_projection_vs_y()

if __name__ == "__main__":
    main()
