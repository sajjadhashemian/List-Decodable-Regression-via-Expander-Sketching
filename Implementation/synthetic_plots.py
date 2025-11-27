"""
Plotting helpers for synthetic experiments (t-SNE and projection vs y).
"""

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data import generate_regression_with_outliers


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
    """
    figures_dir = "Figures"
    os.makedirs(figures_dir, exist_ok=True)
    if not os.path.dirname(save_path):
        save_path = os.path.join(figures_dir, save_path)

    X, y, w_star, inlier_mask, info = generate_regression_with_outliers(
        n=n,
        d=d,
        alpha=alpha,
        outlier_mode=outlier_mode,
        outlier_scale=outlier_scale,
        random_state=random_state,
    )

    z = X @ w_star
    inliers = inlier_mask
    outliers = ~inlier_mask

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
