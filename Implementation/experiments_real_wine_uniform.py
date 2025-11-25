# experiments_real_diabetes.py

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import expander_sketch_list_regression


# ---------------------------------------------------------
# Build LDR mixture via uniform label corruption
# ---------------------------------------------------------
def build_label_corruption_mix(
    X_clean,
    y_clean,
    alpha=0.3,
    n_total=300,
    outlier_scale=10.0,
    random_state=0,
):

    rng = np.random.default_rng(random_state)
    n_pool = X_clean.shape[0]

    if n_total > n_pool:
        raise ValueError("Requested n_total larger than pool size.")

    idx_all = rng.choice(n_pool, size=n_total, replace=False)

    n_in = int(round(alpha * n_total))
    n_out = n_total - n_in

    idx_in = idx_all[:n_in]
    idx_out = idx_all[n_in:]

    X_in = X_clean[idx_in]
    y_in = y_clean[idx_in]

    X_out = X_clean[idx_out]

    y_std = max(1e-6, np.std(y_clean))

    y_out = rng.uniform(
        low=-outlier_scale * y_std,
        high=outlier_scale * y_std,
        size=n_out,
    )

    X_mix = np.vstack([X_in, X_out])
    y_mix = np.concatenate([y_in, y_out])
    mask = np.array([1]*n_in + [0]*n_out, dtype=bool)

    perm = rng.permutation(n_total)
    return X_mix[perm], y_mix[perm], mask[perm]


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------
def main():

    # 1. Load DIABETES dataset
    diabetes = load_diabetes()
    X_raw = diabetes.data
    y_raw = diabetes.target

    print(f"Diabetes loaded: n={X_raw.shape[0]}, d={X_raw.shape[1]}")

    # 2. Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # 3. Train/test split
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_std, y_raw,
        test_size=100,
        random_state=123,
    )

    print(f"Clean train pool = {X_pool.shape[0]}")
    print(f"Clean test size  = {X_test.shape[0]}")

    # Oracle OLS (clean)
    ols_oracle = LinearRegression().fit(X_pool, y_pool)
    mse_oracle = mean_squared_error(y_test, ols_oracle.predict(X_test))
    print(f"[Oracle OLS] MSE = {mse_oracle:.4f}")

    # 4. Build LDR mixture
    alpha = 0.30
    n_total = min(300, X_pool.shape[0])
    outlier_scale = 10.0

    X_mix, y_mix, mask = build_label_corruption_mix(
        X_pool, y_pool,
        alpha=alpha,
        n_total=n_total,
        outlier_scale=outlier_scale,
        random_state=456,
    )

    eff_alpha = mask.mean()
    print(f"Mixture: effective alpha = {eff_alpha:.3f}")

    # 5. t-SNE Visualization
    print("Running t-SNE...")
    Z = TSNE(perplexity=30, learning_rate=100, random_state=0).fit_transform(X_mix)

    plt.figure(figsize=(8,6))
    plt.scatter(Z[mask,0], Z[mask,1], c='blue', s=18, label="Inliers")
    plt.scatter(Z[~mask,0], Z[~mask,1], c='red', s=18, label="Outliers")
    plt.legend()
    plt.title("t-SNE of Diabetes Dataset (LDR mixture)")
    plt.tight_layout()
    plt.savefig("tsne_diabetes_inlier_outlier.pdf", dpi=300)
    plt.close()

    # Colored by label
    plt.figure(figsize=(8,6))
    sc = plt.scatter(Z[:,0], Z[:,1], c=y_mix, cmap='viridis', s=20)
    plt.colorbar(sc, label="Label value (after corruption)")
    plt.title("t-SNE of Diabetes Colored by y")
    plt.tight_layout()
    plt.savefig("tsne_diabetes_label_color.pdf", dpi=300)
    plt.close()

    print("Saved t-SNE plots.")

    # 6. Baselines
    ols_mix = LinearRegression().fit(X_mix, y_mix)
    mse_ols = mean_squared_error(y_test, ols_mix.predict(X_test))
    print(f"[OLS mixture] MSE = {mse_ols:.4f}")

    beta_ridge = fit_ridge(X_mix, y_mix, alpha=1.0)
    mse_ridge = mean_squared_error(y_test, X_test @ beta_ridge)
    print(f"[Ridge] MSE = {mse_ridge:.4f}")

    beta_huber = fit_huber(X_mix, y_mix)
    mse_huber = mean_squared_error(y_test, X_test @ beta_huber)
    print(f"[Huber] MSE = {mse_huber:.4f}")

    beta_ts = fit_theilsen(X_mix, y_mix)
    mse_ts = mean_squared_error(y_test, X_test @ beta_ts)
    print(f"[Theil–Sen] MSE = {mse_ts:.4f}")

    beta_ransac = fit_ransac(X_mix, y_mix)
    mse_ransac = mean_squared_error(y_test, X_test @ beta_ransac)
    print(f"[RANSAC] MSE = {mse_ransac:.4f}")

    # 7. Expander-1
    beta_exp1 = expander_sketch_regression_single_seed(
        X_mix, y_mix,
        alpha=eff_alpha,
        B=1000, r=8, dL=2,
        lambda_reg=1e-4,
        random_state=123
    )
    mse_exp1 = mean_squared_error(y_test, X_test @ beta_exp1)
    print(f"[Expander-1] MSE = {mse_exp1:.4f}")

    # 8. Expander-L
    candidates = expander_sketch_list_regression(
        X_mix, y_mix,
        alpha=eff_alpha,
        r=8, B=1000, dL=2, T=7, R=10,
        lambda_reg=1e-3,
        theta=0.1, rho=0.5,
        cluster_radius=0.0,
        random_state=123,
        verbose=False,
    )

    best = min(mean_squared_error(y_test, X_test @ c) for c in candidates)
    print(f"[Expander-L] BEST MSE = {best:.4f}")

    # Summary
    print("\n========= SUMMARY (Diabetes LDR) =========")
    print(f"Oracle OLS   : {mse_oracle:.4f}")
    print(f"OLS mixture  : {mse_ols:.4f}")
    print(f"Ridge        : {mse_ridge:.4f}")
    print(f"Huber        : {mse_huber:.4f}")
    print(f"Theil–Sen    : {mse_ts:.4f}")
    print(f"RANSAC       : {mse_ransac:.4f}")
    print(f"Expander-1   : {mse_exp1:.4f}")
    print(f"Expander-L   : {best:.4f}")


if __name__ == "__main__":
    main()
