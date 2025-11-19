# expander_sketch_single.py
import numpy as np

from expander_sketch_list import (
    _build_signed_buckets,
    _robust_aggregate_Hg,
    _robust_aggregate_matrices,
)


def expander_sketch_regression_single_seed(
    X,
    y,
    alpha: float,
    r: int = 5,
    B: int = None,
    dL: int = 2,
    T: int = 2,
    M: int = None,
    lambda_reg: float = 1e-3,
    theta: float = 0.5,
    rho: float = 0.2,
    delta: float = 0.1,
    B_const: float = 1.0,
    random_state: int = 0,
    prune_mode: str = "paper",
    verbose: bool = False,
):
    """
    Single-seed version of the Expander-Sketch regression (Algorithm 1
    with R = 1 and no clustering).

    Parameters
    ----------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    alpha : float
        Inlier fraction.
    r : int, default=5
        Number of repetitions (hash functions).
    B : int, optional
        Number of buckets per repetition. If None, we use the theoretical
        scaling B ≍ B_const * (d / alpha) * log(d / delta).
    dL : int, default=2
        Left degree (number of buckets per sample).
    T : int, default=2
        Number of filtering rounds.
    M : int, optional
        Number of MoM blocks for (H, g) aggregation. If None, we use
        max(5, K // 10) where K is the number of active buckets.
    lambda_reg : float, default=1e-3
        Ridge regularization parameter.
    theta : float, default=0.5
        Eigenvalue threshold: continue filtering only if
            lambda_max(C_hat) > (1 + theta) * mean_eig(C_hat).
    rho : float, default=0.2
        Fraction of buckets to prune on each filtering round.
    delta : float, default=0.1
        Failure probability used only in the choice of B when B is None.
    B_const : float, default=1.0
        Constant factor in the theoretical formula for B.
    random_state : int, default=0
        RNG seed.
    prune_mode : {"paper", "flip"}, default="paper"
        "paper": prune buckets with the *highest* Rayleigh scores
                 (as in Algorithm 1 / Lemma 10).
        "flip" : experimental variant that prunes the *lowest* scores.
    verbose : bool, default=False
        If True, print debug information about the number of active buckets.

    Returns
    -------
    beta_hat : ndarray, shape (d,)
        Final regression estimate for this single seed.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # --- Choose B if not provided (faithful to theory up to constants). ---
    if B is None:
        alpha_eff = max(alpha, 1e-3)
        d_eff = max(d, 2)
        delta_eff = min(max(delta, 1e-6), 0.5)
        log_term = np.log(d_eff / delta_eff)
        B = max(10, int(np.ceil(B_const * (d_eff / alpha_eff) * log_term)))

    # --- Build r signed expander bucketings for this *single* seed. ---
    X_buckets_all = []
    y_buckets_all = []
    for _t in range(r):
        X_buckets, y_buckets, _ = _build_signed_buckets(X, y, B, dL, rng)
        X_buckets_all.append(X_buckets)
        y_buckets_all.append(y_buckets)

    # Active buckets are all (t, b) pairs initially.
    active_pairs = [(t, b) for t in range(r) for b in range(B)]
    beta_hat = None

    if verbose:
        print(f"[single-seed] starting with {len(active_pairs)} active buckets (B={B}, r={r})")

    # ===================== Filtering rounds τ = 0..T =====================
    for tau in range(T + 1):
        # 1) Build the list of (H_{t,b}, g_{t,b}) over active buckets.
        H_list = []
        g_list = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            H_tb = Xb.T @ Xb
            g_tb = Xb.T @ yb
            H_list.append(H_tb)
            g_list.append(g_tb)

        if len(H_list) == 0:
            # Fallback: global ridge regression.
            XtX = X.T @ X
            Xty = X.T @ y
            Sigma_reg = XtX + lambda_reg * np.eye(d)
            beta_hat = np.linalg.solve(Sigma_reg, Xty)
            if verbose:
                print("[single-seed] no active buckets; fell back to global ridge.")
            break

        K = len(H_list)
        if M is None:
            M_eff = max(5, K // 10)
        else:
            M_eff = min(M, K)

        # 2) Robust aggregation of normal equations: (Sigma_hat, g_hat)
        Sigma_hat, g_hat = _robust_aggregate_Hg(H_list, g_list, M_eff, rng)

        # 3) Solve (Sigma_hat + lambda I) beta = g_hat
        Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
        beta_hat = np.linalg.solve(Sigma_reg, g_hat)

        if tau == T:
            # Final round: stop after solving.
            break

        # 4) Residual covariance C_hat via robust aggregation.
        C_list = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta_hat
            w = r_tb ** 2
            C_tb = Xb.T @ (w[:, None] * Xb)   # Aᵀ diag(r²) A
            C_list.append(C_tb)

        if len(C_list) == 0:
            break

        Kc = len(C_list)
        Mc_eff = max(5, Kc // 10)
        C_hat = _robust_aggregate_matrices(C_list, Mc_eff, rng)

        # 5) Top eigenpair of C_hat and eigenvalue test.
        eigvals, eigvecs = np.linalg.eigh(C_hat)
        idx_max = np.argmax(eigvals)
        lambda_max = eigvals[idx_max]
        v = eigvecs[:, idx_max]

        target_var = np.mean(eigvals)
        if lambda_max <= (1.0 + theta) * target_var:
            # No strong outlier direction ⇒ stop pruning.
            if verbose:
                print(f"[single-seed] τ={tau}: no strong direction, stopping filtering.")
            break

        # 6) Score and prune buckets.
        scored_pairs = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta_hat
            w = r_tb ** 2
            Xv = Xb @ v
            score_tb = np.sum(w * (Xv ** 2))  # Rayleigh quotient vᵀ C_tb v
            scored_pairs.append(((t, b), score_tb))

        if len(scored_pairs) == 0:
            break

        if prune_mode == "paper":
            # Algorithm 1 / Lemma 10: prune largest scores.
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
        elif prune_mode == "flip":
            # Experimental: prune smallest scores.
            scored_pairs.sort(key=lambda x: x[1])
        else:
            raise ValueError(f"Unknown prune_mode: {prune_mode}")

        k_prune = max(1, int(np.floor(rho * len(scored_pairs))))
        to_prune = set(pair for (pair, _) in scored_pairs[:k_prune])

        active_pairs = [pair for pair in active_pairs if pair not in to_prune]
        if verbose:
            print(f"[single-seed] τ={tau}: pruned {k_prune}, active={len(active_pairs)}")

        if len(active_pairs) == 0:
            break

    return beta_hat
