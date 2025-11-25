# experiments_real_casp_concrete_tsne.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from baselines_sklearn import fit_ridge, fit_huber, fit_ransac, fit_theilsen
from expander_sketch_single import expander_sketch_regression_single_seed
from expander_sketch_list import expander_sketch_list_regression


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_casp(path="Datasets/CASP/CASP.csv"):
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy(float)
    X = df.iloc[:, 1:].to_numpy(float)
    return X, y

def load_concrete(path="Datasets/concrete+compressive+strength/Concrete_Data.xls"):
    df = pd.read_excel(path)
    y = df.iloc[:, -1].to_numpy(float)
    X = df.iloc[:, :-1].to_numpy(float)
    return X, y


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def joint_preprocess_to_10d(X_in, X_out, degree=2, d_target=10, random_state=0):
    d_in = X_in.shape[1]
    d_out = X_out.shape[1]

    if d_in != d_out:
        d_max = max(d_in, d_out)
        if d_in < d_max:
            X_in = np.hstack([X_in, np.zeros((X_in.shape[0], d_max - d_in))])
        if d_out < d_max:
            X_out = np.hstack([X_out, np.zeros((X_out.shape[0], d_max - d_out))])

    X_all = np.vstack([X_in, X_out])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_std)

    pca = PCA(n_components=d_target, random_state=random_state)
    X_10d = pca.fit_transform(X_poly)

    n_in = X_in.shape[0]
    return X_10d[:n_in], X_10d[n_in:]


# ---------------------------------------------------------
# Real-data mixture builder
# ---------------------------------------------------------

def build_real_mix(
    X_in, y_in,
    X_out, y_out,
    alpha=0.30,
    n_total=1400,
    random_state=0,
):
    rng = np.random.default_rng(random_state)

    n_in_total = X_in.shape[0]
    n_out_total = X_out.shape[0]

    n_in = int(alpha * n_total)
    n_out = n_total - n_in

    idx_in = rng.choice(n_in_total, n_in, replace=False)
    idx_out = rng.choice(n_out_total, n_out, replace=False)

    X_in_sub = X_in[idx_in]
    y_in_sub = y_in[idx_in]

    X_out_sub = X_out[idx_out]
    y_out_sub = y_out[idx_out]

    # Permute outlier labels
    y_out_perm = rng.permutation(y_out_sub)

    X_mix = np.vstack([X_in_sub, X_out_sub])
    y_mix = np.concatenate([y_in_sub, y_out_perm])
    mask = np.array([1]*n_in + [0]*n_out, bool)

    perm = rng.permutation(n_total)
    return X_mix[perm], y_mix[perm], mask[perm]


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    # Load datasets
    X_casp_raw, y_casp = load_casp()
    X_conc_raw, y_conc = load_concrete()

    print("CASP:", X_casp_raw.shape)
    print("Concrete:", X_conc_raw.shape)

    # --------------------------------------------
    # Preprocess to common 10D space
    # --------------------------------------------
    X_casp_10, X_conc_10 = joint_preprocess_to_10d(X_casp_raw, X_conc_raw)

    # --------------------------------------------
    # t-SNE PLOT — added here
    # --------------------------------------------
    print("Running t-SNE...")
    X_all = np.vstack([X_casp_10, X_conc_10])
    labels = np.array([1]*len(X_casp_10) + [0]*len(X_conc_10))  # 1=inlier, 0=outlier

    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=0)
    X_emb = tsne.fit_transform(X_all)

    plt.figure(figsize=(8,6))
    plt.scatter(X_emb[labels==1,0], X_emb[labels==1,1], s=12, alpha=0.8, label="CASP Inliers", c='tab:blue')
    plt.scatter(X_emb[labels==0,0], X_emb[labels==0,1], s=12, alpha=0.8, label="Concrete Outliers", c='tab:orange')
    plt.legend()
    plt.title("t-SNE of CASP (inliers) + Concrete (outliers)")
    plt.tight_layout()
    plt.savefig("tsne_casp_concrete.pdf")
    print("Saved t-SNE to tsne_casp_concrete.pdf\n")

    # --------------------------------------------
    # Build CASP-only clean test set
    # --------------------------------------------
    rng = np.random.default_rng(123)
    n_test = 1000
    idx_test = rng.choice(len(X_casp_10), size=n_test, replace=False)
    mask_test = np.zeros(len(X_casp_10), bool)
    mask_test[idx_test] = True

    X_test = X_casp_10[mask_test]
    y_test = y_casp[mask_test]

    X_casp_train = X_casp_10[~mask_test]
    y_casp_train = y_casp[~mask_test]

    # Oracle CASP-only
    ols_oracle = LinearRegression().fit(X_casp_train, y_casp_train)
    mse_oracle = mean_squared_error(y_test, ols_oracle.predict(X_test))
    print("Oracle OLS =", mse_oracle)

    # --------------------------------------------
    # Build mixture CASP-inliers + Concrete-outliers
    # --------------------------------------------
    X_mix, y_mix, mask = build_real_mix(
        X_in=X_casp_train, y_in=y_casp_train,
        X_out=X_conc_10, y_out=y_conc,
        alpha=0.30,
        n_total=1400,
        random_state=456
    )

    eff_alpha = mask.mean()
    print("Effective α =", eff_alpha)

    # --------------------------------------------
    # Run regressors
    # --------------------------------------------

    # Baselines
    mse_ols = mean_squared_error(y_test, LinearRegression().fit(X_mix, y_mix).predict(X_test))
    mse_ridge = mean_squared_error(y_test, X_test @ fit_ridge(X_mix, y_mix, alpha=1.0))
    mse_huber = mean_squared_error(y_test, X_test @ fit_huber(X_mix, y_mix))
    mse_ts = mean_squared_error(y_test, X_test @ fit_theilsen(X_mix, y_mix))
    mse_ransac = mean_squared_error(y_test, X_test @ fit_ransac(X_mix, y_mix))

    print("OLS mixture =", mse_ols)
    print("Ridge =", mse_ridge)
    print("Huber =", mse_huber)
    print("Theil–Sen =", mse_ts)
    print("RANSAC =", mse_ransac)

    # Expander-1
    beta1 = expander_sketch_regression_single_seed(
        X_mix, y_mix,
        alpha=eff_alpha, B=1000, r=8, dL=2, lambda_reg=1e-4, random_state=123
    )
    mse_exp1 = mean_squared_error(y_test, X_test @ beta1)
    print("Expander-1 =", mse_exp1)

    # Expander-L
    candidates = expander_sketch_list_regression(
        X_mix, y_mix,
        alpha=eff_alpha,
        r=7, B=1000, dL=2, T=10, R=15,
        lambda_reg=1e-3,
        theta=0.07, rho=0.45,
        cluster_radius=0.0,
        random_state=123,
    )

    mse_expL = min(mean_squared_error(y_test, X_test @ c) for c in candidates)
    print("Expander-L best =", mse_expL)


if __name__ == "__main__":
    main()
