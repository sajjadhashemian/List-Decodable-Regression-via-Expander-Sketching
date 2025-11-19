# expander_sketch_list.py
import numpy as np
# could use geometeric_median instead of MoM
# from sklearn.covariance import geometric_median

def _build_signed_buckets(X, y, B, dL, rng):
    """
    Build signed expander-style buckets for one repetition t.

    For each sample i in [n], choose dL distinct buckets in [B] and
    assign independent Rademacher signs in {+1, -1}. For each edge (i -> b)
    with sign sigma, we append sigma * x_i as a row in bucket b and sigma * y_i
    as the response, and we record the original index i.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    B : int
        Number of buckets.
    dL : int
        Left degree (number of buckets per sample).
    rng : np.random.Generator

    Returns
    -------
    X_buckets : list of length B
        Each entry is either None or an array of shape (n_b, d).
    y_buckets : list of length B
        Each entry is either None or an array of shape (n_b,).
    idx_buckets : list of length B
        Each entry is either None or an int array of shape (n_b,) recording
        the original sample indices.
    """
    n, d = X.shape
    X_lists = [[] for _ in range(B)]
    y_lists = [[] for _ in range(B)]
    idx_lists = [[] for _ in range(B)]

    for i in range(n):
        buckets_i = rng.choice(B, size=dL, replace=False)
        signs_i = rng.choice(np.array([-1.0, 1.0]), size=dL)
        xi = X[i]
        yi = y[i]
        for b, sgn in zip(buckets_i, signs_i):
            X_lists[b].append(sgn * xi)
            y_lists[b].append(sgn * yi)
            idx_lists[b].append(i)

    X_buckets = []
    y_buckets = []
    idx_buckets = []
    for b in range(B):
        if len(X_lists[b]) == 0:
            X_buckets.append(None)
            y_buckets.append(None)
            idx_buckets.append(None)
        else:
            X_buckets.append(np.vstack(X_lists[b]))
            y_buckets.append(np.array(y_lists[b]))
            idx_buckets.append(np.array(idx_lists[b], dtype=int))

    return X_buckets, y_buckets, idx_buckets

def _robust_aggregate_Hg(H_list, g_list, M, rng):
    """
    Median-of-means aggregation over (H, g) pairs.

    Parameters
    ----------
    H_list : list of np.ndarray, each (d, d)
    g_list : list of np.ndarray, each (d,)
    M : int
        Number of MoM blocks.
    rng : np.random.Generator

    Returns
    -------
    Sigma_hat : (d, d)
    g_hat : (d,)
    """
    K = len(H_list)
    if K == 0:
        raise ValueError("No bucket statistics to aggregate")

    H_arr = np.stack(H_list, axis=0)  # (K, d, d)
    g_arr = np.stack(g_list, axis=0)  # (K, d)

    indices = np.arange(K)
    rng.shuffle(indices)
    blocks = np.array_split(indices, M)

    H_blocks = []
    g_blocks = []
    for blk in blocks:
        if len(blk) == 0:
            continue
        H_mean = np.mean(H_arr[blk], axis=0)
        g_mean = np.mean(g_arr[blk], axis=0)
        H_blocks.append(H_mean)
        g_blocks.append(g_mean)

    H_blocks = np.stack(H_blocks, axis=0)  # (M', d, d)
    g_blocks = np.stack(g_blocks, axis=0)  # (M', d)

    Sigma_hat = np.median(H_blocks, axis=0)
    g_hat = np.median(g_blocks, axis=0)

    return Sigma_hat, g_hat

def _robust_aggregate_matrices(M_list, M, rng):
    """
    Median-of-means aggregation over a list of d x d matrices.

    Used for the residual covariance C_hat = RobustAgg( Aᵀ diag(r²) A ).

    Parameters
    ----------
    M_list : list of np.ndarray, each (d, d)
    M : int
        Number of MoM blocks.

    Returns
    -------
    M_hat : (d, d)
    """
    K = len(M_list)
    if K == 0:
        raise ValueError("No matrices to aggregate")

    M_arr = np.stack(M_list, axis=0)  # (K, d, d)
    indices = np.arange(K)
    rng.shuffle(indices)
    blocks = np.array_split(indices, M)

    M_blocks = []
    for blk in blocks:
        if len(blk) == 0:
            continue
        M_mean = np.mean(M_arr[blk], axis=0)
        M_blocks.append(M_mean)

    M_blocks = np.stack(M_blocks, axis=0)  # (M', d, d)
    M_hat = np.median(M_blocks, axis=0)
    return M_hat

def _build_all_seeds_buckets(X, y, alpha, r, dL, R, rng_global, B=None, delta: float = 0.1, B_const: float = 1.0):
    """
    For each seed s in {1, ..., R}, build r signed expander sketches.

    Each seed s yields:
      - r repetitions (t = 0..r-1)
      - B buckets per repetition
    using signed expander-style bucketings.

    Parameters
    ----------
    X, y : data and responses.
    alpha : float
        Inlier fraction, used only to choose B if not given.
    r : int
        Number of repetitions.
    dL : int
        Left degree.
    R : int
        Number of independent seeds.
    rng_global : np.random.Generator
    B : int, optional
        Number of buckets per repetition. If None, use B ≈ B_const * (d/alpha) * log(d/delta).

    Returns
    -------
    seeds_info : list of dict
        Each dict has keys:
            "X_buckets": list length r, each an object of length B of bucket matrices
            "y_buckets": same structure
            "idx_buckets": same structure (original indices for each row)
            "B": scalar, number of buckets
    """
    n, d = X.shape
    seeds_info = []

    if B is None:
        alpha_eff = max(alpha, 1e-3)
        d_eff = max(d, 2)
        delta_eff = min(max(delta, 1e-6), 0.5)  # avoid weird values

    # Theoretical scaling:  B ≍ B_const * (d/alpha) * log(d/delta)
        log_term = np.log(d_eff / delta_eff)
        B = max(10, int(np.ceil(B_const * (d_eff / alpha_eff) * log_term)))


    for _ in range(R):
        rng = np.random.default_rng(rng_global.integers(1, 10**9))
        seed_X_buckets = []
        seed_y_buckets = []
        seed_idx_buckets = []

        for _t in range(r):
            X_buckets, y_buckets, idx_buckets = _build_signed_buckets(X, y, B, dL, rng)
            seed_X_buckets.append(X_buckets)
            seed_y_buckets.append(y_buckets)
            seed_idx_buckets.append(idx_buckets)

        seeds_info.append(
            {
                "X_buckets": seed_X_buckets,
                "y_buckets": seed_y_buckets,
                "idx_buckets": seed_idx_buckets,
                "B": B,
            }
        )

    return seeds_info

def expander_sketch_list_regression(
    X,
    y,
    alpha: float,
    r: int = 5,
    B: int = None,
    dL: int = 2,
    T: int = 2,
    R: int = 10,
    M: int = None,
    lambda_reg: float = 1e-3,
    theta: float = 0.5,
    rho: float = 0.2,
    delta=0.1, 
    B_const=1.0,
    cluster_radius: float = 0.0, # not specified in the algorithm
    random_state: int = 0,
    prune_mode: str = "paper",
    verbose: bool = False,
):
    """
    Multi-seed, list-producing expander-sketch regression (Algorithm 1 style).

    For each seed s in {1, ..., R}:
      - Build r signed expander sketches (buckets) of [n].
      - Run T rounds (τ = 0..T):
          * Compute bucket-wise normal equations (H_{t,b}, g_{t,b}) on active buckets.
          * Robustly aggregate these via MoM to get (Σ̂_τ, ĝ_τ).
          * Solve (Σ̂_τ + λI) β_τ = ĝ_τ.
          * If τ < T, perform residual-based spectral filtering:
              - Robustly aggregate residual covariances.
              - Take top eigenvector v of Ĉ.
              - Score each bucket by vᵀ Aᵀ diag(r²) A v.
              - Remove a ρ-fraction of buckets according to prune_mode.

      - Store the final β_s from this seed.

    Finally, cluster the {β_s} using a simple radius-based heuristic
    and return the cluster centers as the candidate list.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    alpha : float
        Inlier fraction, used in the choice of B if not provided.
    r : int, default=5
        Number of repetitions per seed.
    B : int, optional
        Number of buckets per repetition.
    dL : int, default=2
        Left degree.
    T : int, default=2
        Number of filtering rounds.
    R : int, default=10
        Number of seeds.
    M : int, optional
        Number of MoM blocks for (H, g) aggregation.
    lambda_reg : float, default=1e-3
        Ridge parameter.
    theta : float, default=0.5
        Threshold for deciding whether there is a strong outlier direction:
        require λ_max(Ĉ) > (1 + η) * mean_eig(Ĉ) to continue pruning.
    rho : float, default=0.2
        Fraction of buckets to prune each round.
    cluster_radius : float, default=5.0
        Radius used in simple clustering of β_s.
    random_state : int, default=0
        RNG seed.
    prune_mode : {"paper", "flip"}, default="paper"
        "paper": prune highest scores (as stated in Algorithm 1 / Lemma 10).
        "flip" : experimental variant that prunes lowest scores.

    Returns
    -------
    centers : list of ndarray, each shape (d,)
        Candidate regression vectors.
    """
    rng_global = np.random.default_rng(random_state)
    n, d = X.shape

    seeds_info = _build_all_seeds_buckets(X, y, alpha, r, dL, R, rng_global, B=B, delta=delta, B_const=B_const)
    candidates = []

    for seed in seeds_info:
        idx_buckets_all = seed["idx_buckets"]  # currently unused but useful for debugging
        B_val = seed["B"]
        X_buckets_all = seed["X_buckets"]  # length r
        y_buckets_all = seed["y_buckets"]

        # Active buckets: list of (t, b) pairs
        active_pairs = [(t, b) for t in range(r) for b in range(B_val)]
        beta_s = None

        if verbose:
            print(f"Seed: starting with {len(active_pairs)} active buckets")

        for tau in range(T + 1):
            # 1) Collect (H_{t,b}, g_{t,b}) from active buckets
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
                # Fallback: global ridge
                XtX = X.T @ X
                Xty = X.T @ y
                Sigma_reg = XtX + lambda_reg * np.eye(d)
                beta_s = np.linalg.solve(Sigma_reg, Xty)
                break

            K = len(H_list)
            if M is None:
                M_eff = max(5, K // 10)
            else:
                M_eff = min(M, K)

            Sigma_hat, g_hat = _robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)

            # 3) Solve (Sigma_hat + lambda I) beta = g_hat
            Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
            beta_s = np.linalg.solve(Sigma_reg, g_hat)

            if tau == T:
                # No more filtering rounds
                break

            # 4) Residual covariance via robust aggregation
            C_list = []
            for (t, b) in active_pairs:
                Xb = X_buckets_all[t][b]
                yb = y_buckets_all[t][b]
                if Xb is None or yb is None or len(yb) == 0:
                    continue
                r_tb = yb - Xb @ beta_s
                w = r_tb ** 2
                C_tb = Xb.T @ (w[:, None] * Xb)
                C_list.append(C_tb)

            if len(C_list) == 0:
                break

            # RobustAggregation over residual covariances
            Kc = len(C_list)
            Mc_eff = max(5, Kc // 10)
            C_hat = _robust_aggregate_matrices(C_list, Mc_eff, rng_global)

            # Top eigenpair of C_hat
            eigvals, eigvecs = np.linalg.eigh(C_hat)
            idx_max = np.argmax(eigvals)
            lambda_max = eigvals[idx_max]
            v = eigvecs[:, idx_max]

            target_var = np.mean(eigvals)

            # If no strong outlier direction, stop filtering
            if lambda_max <= (1.0 + theta) * target_var:
                break

            # Score buckets by vᵀ Aᵀ diag(r²) A v
            scored_pairs = []
            for (t, b) in active_pairs:
                Xb = X_buckets_all[t][b]
                yb = y_buckets_all[t][b]
                if Xb is None or yb is None or len(yb) == 0:
                    continue
                r_tb = yb - Xb @ beta_s
                w = r_tb ** 2
                Xv = Xb @ v
                score_tb = np.sum(w * (Xv ** 2))  # vᵀ C_tb v
                scored_pairs.append(((t, b), score_tb))

            if len(scored_pairs) == 0:
                break

            # Decide pruning direction based on prune_mode
            if prune_mode == "paper":
                # Algorithm 1 / Lemma 10: prune highest scores
                scored_pairs.sort(key=lambda x: x[1], reverse=True)
            elif prune_mode == "flip":
                # Experimental variant: prune lowest scores
                scored_pairs.sort(key=lambda x: x[1])
            else:
                raise ValueError(f"Unknown prune_mode: {prune_mode}")

            k_prune = max(1, int(np.floor(rho * len(scored_pairs))))
            to_prune = set(pair for (pair, _) in scored_pairs[:k_prune])

            active_pairs = [pair for pair in active_pairs if pair not in to_prune]
            if verbose:
                print(f"  After tau={tau} pruning, active buckets = {len(active_pairs)}")

            if len(active_pairs) == 0:
                break

        candidates.append(beta_s)

    # --- Simple radius-based clustering of candidates ---
    centers = []
    for beta in candidates:
        if beta is None:
            continue
        if len(centers) == 0:
            centers.append(beta)
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

# Theoretical potential and debugging
def oracle_inlier_bucket_regression(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = 5,
    B: int = 1000,
    dL: int = 2,
    k_in_min: int = 5,
    k_out_max: int = 5,
    lambda_reg: float = 1e-3,
    random_state: int = 0,
):
    """
    Oracle version of the expander-based regression:
    - Uses the same signed buckets as Algorithm 1
    - BUT keeps only buckets that are 'good':
        #inliers >= k_in_min and #outliers <= k_out_max
      based on the true inlier_mask.

    This is for debugging theoretical potential:
    it shows how well we *could* do if filtering were perfect.
    """
    rng_global = np.random.default_rng(random_state)
    seeds_info = _build_all_seeds_buckets(X, y, alpha, r, dL, R=1, rng_global=rng_global, B=B)
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    n, d = X.shape

    # Collect H, g only from 'good' buckets
    H_list = []
    g_list = []

    for t in range(r):
        for b in range(B_val):
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            idxs = idx_buckets_all[t][b]
            if Xb is None or yb is None or idxs is None or len(yb) == 0:
                continue

            inliers_b = np.sum(inlier_mask[idxs])
            outliers_b = len(idxs) - inliers_b

            # Keep only buckets with sufficient inliers and small outlier count
            if inliers_b >= k_in_min and outliers_b <= k_out_max:
                H_tb = Xb.T @ Xb
                g_tb = Xb.T @ yb
                H_list.append(H_tb)
                g_list.append(g_tb)

    if len(H_list) == 0:
        print("Oracle: no 'good' buckets found under given thresholds.")
        # fallback: global ridge
        XtX = X.T @ X
        Xty = X.T @ y
        Sigma_reg = XtX + lambda_reg * np.eye(d)
        beta_oracle = np.linalg.solve(Sigma_reg, Xty)
        return beta_oracle, 0, 0

    K = len(H_list)
    M_eff = max(5, K // 10)
    rng = np.random.default_rng(random_state + 1)
    Sigma_hat, g_hat = _robust_aggregate_Hg(H_list, g_list, M_eff, rng)

    Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
    beta_oracle = np.linalg.solve(Sigma_reg, g_hat)

    return beta_oracle, K, B_val

def debug_bucket_scores_vs_goodness(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = 5,
    B: int = 1000,
    dL: int = 2,
    lambda_reg: float = 1e-3,
    random_state: int = 0,
    k_in_min: int = 5,
    k_out_max: int = 5,
):
    """
    Debug: for a single seed, compute the usual spectral direction v and
    the Rayleigh scores score_tb = sum_i w_i * (Xb v)^2, and compare these
    scores between 'good' and 'bad' buckets (based on inlier_mask).
    """
    rng_global = np.random.default_rng(random_state)
    seeds_info = _build_all_seeds_buckets(X, y, alpha, r, dL, R=1, rng_global=rng_global, B=B)
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    n, d = X.shape

    # First, compute a beta using all buckets (like one tau=0 step)
    H_list = []
    g_list = []
    active_pairs = [(t, b) for t in range(r) for b in range(B_val)]
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
        print("Debug scores: no bucket stats available.")
        return

    K = len(H_list)
    M_eff = max(5, K // 10)
    Sigma_hat, g_hat = _robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)
    Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
    beta = np.linalg.solve(Sigma_reg, g_hat)

    # Compute residual covariance C_hat (same as algorithm)
    C_list = []
    for (t, b) in active_pairs:
        Xb = X_buckets_all[t][b]
        yb = y_buckets_all[t][b]
        if Xb is None or yb is None or len(yb) == 0:
            continue
        r_tb = yb - Xb @ beta
        w = r_tb ** 2
        C_tb = Xb.T @ (w[:, None] * Xb)
        C_list.append(C_tb)

    if len(C_list) == 0:
        print("Debug scores: no C_tb available.")
        return

    C_hat = _robust_aggregate_matrices(C_list, max(5, len(C_list) // 10), rng_global)
    eigvals, eigvecs = np.linalg.eigh(C_hat)
    idx_max = np.argmax(eigvals)
    v = eigvecs[:, idx_max]

    # Now compute score_tb and label buckets as good / bad
    scores_good = []
    scores_bad = []

    for (t, b) in active_pairs:
        Xb = X_buckets_all[t][b]
        yb = y_buckets_all[t][b]
        idxs = idx_buckets_all[t][b]
        if Xb is None or yb is None or idxs is None or len(yb) == 0:
            continue

        inliers_b = np.sum(inlier_mask[idxs])
        outliers_b = len(idxs) - inliers_b

        r_tb = yb - Xb @ beta
        w = r_tb ** 2
        Xv = Xb @ v
        score_tb = np.sum(w * (Xv ** 2))

        is_good = (inliers_b >= k_in_min) and (outliers_b <= k_out_max)
        if is_good:
            scores_good.append(score_tb)
        else:
            scores_bad.append(score_tb)

    if len(scores_good) == 0:
        print("Debug scores: no 'good' buckets under thresholds.")
        return

    scores_good = np.array(scores_good)
    scores_bad = np.array(scores_bad)

    print(f"Debug scores: #good buckets = {len(scores_good)}, #bad buckets = {len(scores_bad)}")
    print(f"  mean(score_good) = {scores_good.mean():.4e}, mean(score_bad) = {scores_bad.mean():.4e}")
    print(f"  median(score_good) = {np.median(scores_good):.4e}, median(score_bad) = {np.median(scores_bad):.4e}")

    # Fraction of good buckets among lowest / highest score deciles
    all_scores = np.concatenate([scores_good, scores_bad])
    threshold_low = np.quantile(all_scores, 0.1)
    threshold_high = np.quantile(all_scores, 0.9)

    frac_good_low = np.mean(scores_good <= threshold_low)
    frac_good_high = np.mean(scores_good >= threshold_high)

    print(f"  Fraction of GOOD buckets in lowest 10% scores: {frac_good_low:.3f}")
    print(f"  Fraction of GOOD buckets in highest 10% scores: {frac_good_high:.3f}")

def debug_survivor_goodness(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = 8,
    B: int = 1000,
    dL: int = 2,
    T: int = 7,
    lambda_reg: float = 1e-3,
    theta: float = 0.1,
    rho: float = 0.5,
    random_state: int = 123,
    prune_mode: str = "flip",
    k_in_min: int = 5,
    k_out_max: int = 5,
):
    """
    Run a single-seed version of the expander-sketch list algorithm,
    and at the end report how many *surviving* buckets are 'good'
    (>= k_in_min inliers and <= k_out_max outliers).

    This tells us how well the filtering isolates good buckets.
    """
    rng_global = np.random.default_rng(random_state)

    # Build buckets for a single seed
    seeds_info = _build_all_seeds_buckets(
        X, y, alpha, r, dL, R=1, rng_global=rng_global, B=B
    )
    seed = seeds_info[0]

    B_val = seed["B"]
    X_buckets_all = seed["X_buckets"]
    y_buckets_all = seed["y_buckets"]
    idx_buckets_all = seed["idx_buckets"]

    n, d = X.shape

    # Start with all buckets active
    active_pairs = [(t, b) for t in range(r) for b in range(B_val)]
    print(f"[debug_survivor_goodness] Seed: starting with {len(active_pairs)} active buckets")

    beta = None

    for tau in range(T + 1):
        # 1) Build H_list, g_list from active buckets
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
            print("[debug_survivor_goodness] No active buckets with data.")
            return

        K = len(H_list)
        M_eff = max(5, K // 10)
        Sigma_hat, g_hat = _robust_aggregate_Hg(H_list, g_list, M_eff, rng_global)

        Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
        beta = np.linalg.solve(Sigma_reg, g_hat)

        if tau == T:
            # Stop before further filtering
            break

        # 2) Compute residual covariance C_hat just like in main algorithm
        C_list = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta
            w = r_tb ** 2
            C_tb = Xb.T @ (w[:, None] * Xb)
            C_list.append(C_tb)

        if len(C_list) == 0:
            print("[debug_survivor_goodness] No C_tb matrices.")
            return

        C_hat = _robust_aggregate_matrices(C_list, max(5, len(C_list) // 10), rng_global)
        eigvals, eigvecs = np.linalg.eigh(C_hat)
        idx_max = np.argmax(eigvals)
        lambda_max = eigvals[idx_max]
        v = eigvecs[:, idx_max]

        target_var = np.mean(eigvals)
        if lambda_max <= (1.0 + theta) * target_var:
            print(f"[debug_survivor_goodness] tau={tau}: no strong outlier direction, stopping filtering.")
            break

        # 3) Score buckets and prune according to prune_mode
        scored_pairs = []
        for (t, b) in active_pairs:
            Xb = X_buckets_all[t][b]
            yb = y_buckets_all[t][b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            r_tb = yb - Xb @ beta
            w = r_tb ** 2
            Xv = Xb @ v
            score_tb = np.sum(w * (Xv ** 2))
            scored_pairs.append(((t, b), score_tb))

        if len(scored_pairs) == 0:
            print("[debug_survivor_goodness] No scored buckets.")
            return

        if prune_mode == "paper":
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
        elif prune_mode == "flip":
            scored_pairs.sort(key=lambda x: x[1])
        else:
            raise ValueError(f"Unknown prune_mode: {prune_mode}")

        k_prune = max(1, int(np.floor(rho * len(scored_pairs))))
        to_prune = set(pair for (pair, _) in scored_pairs[:k_prune])
        active_pairs = [pair for pair in active_pairs if pair not in to_prune]

        print(f"[debug_survivor_goodness] After tau={tau} pruning, active buckets = {len(active_pairs)}")

        if len(active_pairs) == 0:
            print("[debug_survivor_goodness] All buckets pruned.")
            return

    # After filtering, measure how many surviving buckets are 'good'
    good_survivors = 0
    bad_survivors = 0

    for (t, b) in active_pairs:
        idxs = idx_buckets_all[t][b]
        if idxs is None or len(idxs) == 0:
            continue
        inliers_b = np.sum(inlier_mask[idxs])
        outliers_b = len(idxs) - inliers_b
        is_good = (inliers_b >= k_in_min) and (outliers_b <= k_out_max)
        if is_good:
            good_survivors += 1
        else:
            bad_survivors += 1

    total_survivors = good_survivors + bad_survivors
    if total_survivors == 0:
        print("[debug_survivor_goodness] No surviving buckets with indices.")
        return

    frac_good = good_survivors / total_survivors
    print(f"[debug_survivor_goodness] Final survivors: {total_survivors} buckets")
    print(f"  good survivors = {good_survivors}, bad survivors = {bad_survivors}")
    print(f"  fraction good among survivors = {frac_good:.3f}")

def debug_bucket_contamination(
    X,
    y,
    inlier_mask,
    alpha: float,
    r: int = 5,
    B: int = None,
    dL: int = 2,
    R: int = 1,
    random_state: int = 0,
):
    """
    Debug helper: build signed buckets and print statistics about
    how many inliers / outliers land in each bucket, to see if
    the expander sketches are giving 'lightly contaminated' buckets.

    This does NOT affect the algorithm; it's only for diagnostic use.
    """
    rng_global = np.random.default_rng(random_state)
    seeds_info = _build_all_seeds_buckets(X, y, alpha, r, dL, R, rng_global, B=B)
    seed = seeds_info[0]  # just look at the first seed

    B_val = seed["B"]
    idx_buckets_all = seed["idx_buckets"]

    inlier_counts = []
    outlier_counts = []

    for t in range(r):
        for b in range(B_val):
            idxs = idx_buckets_all[t][b]
            if idxs is None:
                continue
            inliers_b = np.sum(inlier_mask[idxs])
            outliers_b = len(idxs) - inliers_b
            inlier_counts.append(inliers_b)
            outlier_counts.append(outliers_b)

    inlier_counts = np.array(inlier_counts)
    outlier_counts = np.array(outlier_counts)

    print(f"Total buckets with data: {len(inlier_counts)}")
    print(f"Avg inliers per bucket: {inlier_counts.mean():.2f}")
    print(f"Avg outliers per bucket: {outlier_counts.mean():.2f}")

    # Some quantiles
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        i_q = np.quantile(inlier_counts, q)
        o_q = np.quantile(outlier_counts, q)
        print(f"Quantile {q:.2f}: inliers {i_q:.1f}, outliers {o_q:.1f}")

    # Rough 'lightly contaminated' buckets: at least 5 inliers and at most 5 outliers (just a debug threshold)
    good = (inlier_counts >= 5) & (outlier_counts <= 5)
    print(f"Buckets with >=5 inliers and <=5 outliers: {good.sum()} ({100.0 * good.mean():.1f}%)")
