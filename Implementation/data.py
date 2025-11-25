# data.py
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class RegressionDataInfo:
    n: int
    d: int
    alpha: float
    sigma_inlier: float
    outlier_mode: str
    outlier_scale: float
    corrupt_X: bool
    random_state: int
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray

    def to_dict(self):
        d = asdict(self)
        d["inlier_indices"] = self.inlier_indices.tolist()
        d["outlier_indices"] = self.outlier_indices.tolist()
        return d

def generate_regression_with_outliers(
    n: int = 5000,
    d: int = 20,
    alpha: float = 0.3,
    sigma_inlier: float = 0.1,

    # Outlier response noise
    outlier_mode: str = "uniform",   # uniform, gaussian_heavy, skewed, directional, clustered, mixed
    outlier_scale: float = 10.0,

    # Optional leverage outliers (heavy-tailed X)
    corrupt_X: bool = False,
    random_state: int = 123,
    leverage_df: float = 2.0,         # degrees of freedom for heavy-tailed X
):
    """
    Hard synthetic regression data generator (cleaner version):
        - No signflip mode
        - No preset system
        - No anisotropy or normalization options
        - Purely Gaussian design (so we do not destabilize experiments)
        - Strong, realistic, multi-mode outlier corruptions
        - Optional leverage outliers via heavy-tailed X for outliers
    """
    rng = np.random.default_rng(random_state)

    # --- Verify alpha ---
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")

    # --- True regression parameter ---
    w_star = rng.normal(size=d)

    # --- Design matrix (Gaussian isotropic only!) ---
    X = rng.normal(size=(n, d))

    # --- Inlier / outlier split ---
    n_inliers = int(np.round(alpha * n))
    indices = np.arange(n)
    rng.shuffle(indices)
    inlier_idx = indices[:n_inliers]
    outlier_idx = indices[n_inliers:]
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

    # Clean responses
    y_clean = X @ w_star + rng.normal(scale=sigma_inlier, size=n)
    y = y_clean.copy()

    # Leverage outliers: corrupt X with heavy-tailed rows
    m = len(outlier_idx)
    if corrupt_X and m > 0:
        X[outlier_idx] = (
            rng.standard_t(df=leverage_df, size=(m, d)) * outlier_scale
        )

    # Outlier responses
    if m > 0:
        if outlier_mode == "uniform":
            y_out = rng.uniform(-outlier_scale, outlier_scale, size=m)

        elif outlier_mode == "skewed":
            y_out = rng.exponential(scale=outlier_scale, size=m) - (outlier_scale / 2.0)

        elif outlier_mode == "gaussian_heavy":
            y_out = rng.normal(loc=0.0, scale=outlier_scale, size=m)

        elif outlier_mode == "directional":
            # Hard outliers aligned with a random direction
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            y_out = (X[outlier_idx] @ u) * outlier_scale

        elif outlier_mode == "clustered":
            c = rng.normal(loc=0.0, scale=outlier_scale)
            y_out = c + rng.normal(scale=1.0, size=m)

        elif outlier_mode == "mixed":
            # Realistic mixture of several kinds
            modes = rng.choice(
                ["uniform", "gaussian_heavy", "skewed", "directional"],
                size=m,
                replace=True,
            )
            y_out = np.empty(m)
            for j, mode in enumerate(modes):
                if mode == "uniform":
                    y_out[j] = rng.uniform(-outlier_scale, outlier_scale)
                elif mode == "gaussian_heavy":
                    y_out[j] = rng.normal(0.0, outlier_scale)
                elif mode == "skewed":
                    y_out[j] = rng.exponential(outlier_scale) - outlier_scale / 2.0
                elif mode == "directional":
                    u_loc = rng.normal(size=d); u_loc /= np.linalg.norm(u_loc)
                    y_out[j] = (X[outlier_idx[j:j+1]] @ u_loc)[0] * outlier_scale
        else:
            raise ValueError(f"Unknown outlier_mode: {outlier_mode}")

        y[outlier_idx] = y_out

    info = RegressionDataInfo(
        n=n,
        d=d,
        alpha=alpha,
        sigma_inlier=sigma_inlier,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        corrupt_X=corrupt_X,
        random_state=random_state,
        inlier_indices=inlier_idx,
        outlier_indices=outlier_idx,
    )

    return X, y, w_star, inlier_mask, info
