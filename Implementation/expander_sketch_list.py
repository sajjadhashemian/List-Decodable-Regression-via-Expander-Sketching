import numpy as np
from expander_sketch_core import (
    build_signed_buckets,
    robust_aggregate_Hg,
    robust_aggregate_matrices,
    choose_num_buckets,
)

# Utility: apply theory-faithful parameter defaults
def _apply_theory_defaults(alpha, d, r, T, R, dL, B, delta, B_const, use_theory_defaults):
    """
    If use_theory_defaults=True, override parameters according to:

        B  ≍ (d/alpha) * log(d/delta)
        r  ≍ log(1/delta)
        dL = O(1)
        R  ≍ 1/alpha
        T  ≍ log(1/alpha)

    If use_theory_defaults=False, keep user parameters unchanged.
    """
    if not use_theory_defaults:
        return r, T, R, dL, B, delta, B_const

    # Avoid pathological values
    alpha_eff = max(alpha, 1e-4)
    delta_eff = min(max(delta, 1e-6), 0.1)
    d_eff = max(d, 2)

    # r ≍ log(1/delta)
    if r is None:
        r = max(3, int(np.ceil(np.log(1.0 / delta_eff))))

    # T ≍ log(1/alpha)
    if T is None:
        T = max(1, int(np.ceil(np.log(1.0 / alpha_eff))))

    # R ≍ 1/alpha
    if R is None:
        R = max(1, int(np.ceil(1.0 / alpha_eff)))

    # dL = constant degree
    if dL is None:
        dL = 2  # theory: O(1)

    # B ≍ (d/alpha) log(d/delta)
    if B is None:
        B = choose_num_buckets(d_eff, alpha_eff, delta=delta_eff, B_const=B_const)

    return r, T, R, dL, B, delta_eff, B_const

def build_all_seeds_buckets(
    X,
    y,
    alpha,
    r,
    dL,
    R,
    rng_global,
    B=None,
    delta: float = 0.1,
    B_const: float = 1.0,
    use_networkx: bool = False,
    graph=None,
):
    """
    Build signed-bucket sketches for all seeds.

    If use_networkx=False (default): use random hashing (original behavior).
    If use_networkx=True : use the provided NetworkX bipartite graph.
    """
    n, d = X.shape
    seeds_info = []

    if B is None:
        B = choose_num_buckets(d, alpha, delta=delta, B_const=B_const)

    for _ in range(R):
        rng = np.random.default_rng(rng_global.integers(1, 10**9))
        seed_X_buckets = []
        seed_y_buckets = []
        seed_idx_buckets = []

        for _t in range(r):
            X_buckets, y_buckets, idx_buckets = build_signed_buckets(
                X,
                y,
                B,
                dL,
                rng,
                use_networkx=use_networkx,
                graph=graph,
            )
            seed_X_buckets.append(X_buckets)
            seed_y_buckets.append(y_buckets)
            seed_idx_buckets.append(idx_buckets)

        seeds_info.append({
            "X_buckets": seed_X_buckets,
            "y_buckets": seed_y_buckets,
            "idx_buckets": seed_idx_buckets,
            "B": B,
        })

    return seeds_info

# MAIN ALGORITHM 1
def expander_sketch_list_regression(
    X,
    y,
    alpha: float,

    # Theory mode flag
    use_theory_defaults: bool = False,   # <- IMPORTANT: off by default for experiments

    # Algorithmic parameters (can be tuned or swept)
    r: int = None,          # ≍ log(1/delta) in theory
    B: int = None,          # ≍ (d/alpha) * log(d/delta)
    dL: int = None,         # left degree of the expander
    T: int = None,          # ≍ log(1/alpha) in theory
    R: int = None,          # ≍ 1/alpha in theory
    M: int = None,          # MoM blocks -> default computed internally

    # Regularization / Filtering
    lambda_reg: float = 1e-3,
    theta: float = 0.1,
    rho: float = 0.5,

    # Expander sizing parameters
    delta: float = 1e-3,
    B_const: float = 5.0,

    # Clustering
    cluster_radius: float = 0.0,  # no merging of candidate vectors

    # Expander construction mode
    use_networkx: bool = False,
    graph=None,                  # NetworkX graph if use_networkx=True

    # Misc
    random_state: int = 123,
    verbose: bool = False,
):
    """
    Implementation of Algorithm 1 from:
        "List-Decodable Regression via Expander Sketching"

    Modes:
      - use_theory_defaults=True:
          parameters are set using theory-based scaling.
      - use_theory_defaults=False:
          user-specified (tuned / swept) parameters are used.
    """
    rng_global = np.random.default_rng(random_state)
    n, d = X.shape

    # Apply theory-faithful defaults (or leave as-is if flag is False)
    r, T, R, dL, B, delta, B_const = _apply_theory_defaults(
        alpha, d, r, T, R, dL, B, delta, B_const, use_theory_defaults
    )

    # Safety: if user leaves some None with theory defaults off,
    # we can plug in simple reasonable defaults.
    if r is None:
        r = 8
    if T is None:
        T = 7
    if R is None:
        R = 10
    if dL is None:
        dL = 2
    if B is None:
        B = choose_num_buckets(d, alpha, delta=delta, B_const=B_const)

    # Build expander sketches for all seeds
    seeds_info = build_all_seeds_buckets(
        X,
        y,
        alpha,
        r,
        dL,
        R,
        rng_global,
        B=B,
        delta=delta,
        B_const=B_const,
        use_networkx=use_networkx,
        graph=graph,
    )

    candidates = []

    # Main multi-seed loop
    for seed in seeds_info:
        X_buckets_all = seed["X_buckets"]
        y_buckets_all = seed["y_buckets"]
        B_val = seed["B"]

        active_pairs = [(t, b) for t in range(r) for b in range(B_val)]
        beta_s = None

        if verbose:
            print(f"[Seed init] {len(active_pairs)} active buckets")

        # Filtering rounds τ = 0 .. T
        for tau in range(T + 1):

            # Collect all H_tb, g_tb
            H_list, g_list = [], []
            for (t, b) in active_pairs:
                Xb = X_buckets_all[t][b]
                yb = y_buckets_all[t][b]
                if Xb is None or yb is None or len(yb) == 0:
                    continue
                H_list.append(Xb.T @ Xb)
                g_list.append(Xb.T @ yb)

            if len(H_list) == 0:
                # fallback: global ridge
                XtX = X.T @ X
                Xty = X.T @ y
                beta_s = np.linalg.solve(XtX + lambda_reg * np.eye(d), Xty)
                break

            # MoM block size
            K = len(H_list)
            M_eff = max(5, K // 10) if M is None else min(M, K)

            # Aggregate moments
            Sigma_hat, g_hat = robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)
            beta_s = np.linalg.solve(Sigma_hat + lambda_reg * np.eye(d), g_hat)

            if tau == T:
                break

            # Compute residual covariance
            C_list = []
            for (t, b) in active_pairs:
                Xb = X_buckets_all[t][b]
                yb = y_buckets_all[t][b]
                if Xb is None or yb is None or len(yb) == 0:
                    continue
                r_tb = yb - Xb @ beta_s
                C_list.append(Xb.T @ ((r_tb**2)[:, None] * Xb))

            if len(C_list) == 0:
                break

            Kc = len(C_list)
            Mc_eff = max(5, Kc // 10)
            C_hat = robust_aggregate_matrices(C_list, Mc_eff, rng_global)

            # spectral direction
            eigvals, eigvecs = np.linalg.eigh(C_hat)
            idx_max = np.argmax(eigvals)
            lambda_max = eigvals[idx_max]
            v = eigvecs[:, idx_max]
            target_var = np.mean(eigvals)

            if lambda_max <= (1.0 + theta) * target_var:
                break

            # Score buckets
            scored_pairs = []
            for (t, b) in active_pairs:
                Xb = X_buckets_all[t][b]
                yb = y_buckets_all[t][b]
                if Xb is None or yb is None or len(yb) == 0:
                    continue
                r_tb = yb - Xb @ beta_s
                score = np.sum((r_tb**2) * ((Xb @ v) ** 2))
                scored_pairs.append(((t, b), score))

            # prune top rho
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            k_prune = max(1, int(np.floor(rho * len(scored_pairs))))
            to_prune = set(pair for (pair, _) in scored_pairs[:k_prune])
            active_pairs = [pair for pair in active_pairs if pair not in to_prune]

            if verbose:
                print(f"[tau={tau}] active={len(active_pairs)}")

            if len(active_pairs) == 0:
                break

        candidates.append(beta_s)

    # Clustering final list
    centers = []
    for beta in candidates:
        if beta is None:
            continue
        assigned = False
        for j, c in enumerate(centers):
            if np.linalg.norm(beta - c) <= cluster_radius:
                centers[j] = 0.5 * (centers[j] + beta)
                assigned = True
                break
        if not assigned:
            centers.append(beta)

    return centers
