import numpy as np
from expander_sketch_core import (
    build_signed_buckets,
    robust_aggregate_Hg,
    choose_num_buckets,
)

def expander_sketch_regression_single_seed(
    X,
    y,
    alpha: float,

    # Theory mode flag
    use_theory_defaults: bool = False,   # <- off by default

    B: int = None,
    r: int = None,
    dL: int = None,
    M: int = None,
    lambda_reg: float = 1e-3,
    delta: float = 1e-3,
    B_const: float = 5.0,
    random_state: int = 123,

    # Expander construction mode
    use_networkx: bool = False,
    graph=None,
):
    """
    Expander-1: Single-seed version of the expander sketch regression.

    If use_theory_defaults=True:
        r      = ceil(log(1/delta))
        dL     = 2
        B      = choose_num_buckets(d, alpha, delta, B_const)

    If use_theory_defaults=False:
        use the user-specified (tuned / swept) parameters.
    """
    n, d = X.shape
    rng = np.random.default_rng(random_state)

    if use_theory_defaults:
        if r is None:
            r = max(1, int(np.ceil(np.log(1.0 / delta))))   # r ≍ log(1/delta)
        if dL is None:
            dL = 2  # theory: O(1)

    # Simple defaults if the user didn't specify and theory mode is off
    if r is None:
        r = 8
    if dL is None:
        dL = 2
    if B is None:
        B = choose_num_buckets(d, alpha, delta=delta, B_const=B_const)

    # Main Expander-1 computation
    H_list = []
    g_list = []

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

        for b in range(B):
            Xb = X_buckets[b]
            yb = y_buckets[b]
            if Xb is None or yb is None or len(yb) == 0:
                continue
            H_tb = Xb.T @ Xb
            g_tb = Xb.T @ yb
            H_list.append(H_tb)
            g_list.append(g_tb)

    # Fallback: if degeneracy happens, do global ridge
    if len(H_list) == 0:
        XtX = X.T @ X
        Xty = X.T @ y
        Sigma_reg = XtX + lambda_reg * np.eye(d)
        return np.linalg.solve(Sigma_reg, Xty)

    # Number of MoM blocks
    K = len(H_list)
    if M is None:
        M_eff = max(5, K // 10)
    else:
        M_eff = min(M, K)

    Sigma_hat, g_hat = robust_aggregate_Hg(H_list, g_list, M_eff, rng)

    Sigma_reg = Sigma_hat + lambda_reg * np.eye(d)
    beta_hat = np.linalg.solve(Sigma_reg, g_hat)

    return beta_hat
