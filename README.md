# expander-ldr

A research-oriented Python implementation of:

> H. Pourali, S. Hashemian, E. Ardeshir-Larijani,  
> *List-Decodable Regression via Expander Sketching*, arXiv:2511.22524.

The goal is to provide:

- A **scikit-learn compatible** estimator
- A **modular implementation** of all algorithmic pieces (expander sketch, bucket-wise moments, robust aggregation, spectral filtering, and list generation)

---

<!-- ## Installation

```bash
git clone https://github.com/sajjadhashemian/List-Decodable-Regression-via-Expander-Sketching.git
cd expander-ldr
pip install -e .
```

This will install the package `expander_ldr`.

**Requirements:**
- Python >= 3.9
- numpy, scipy, scikit-learn
- matplotlib (for plots in experiments)

--- -->

## Quickstart

```python
import numpy as np
from expander_ldr.estimator import ExpanderLDRRegressor

# Synthetic inlier model
n, d = 5000, 20
alpha = 0.2
rng = np.random.default_rng(0)

X = rng.normal(size=(n, d))
w_star = rng.normal(size=d)
noise = 0.1 * rng.normal(size=n)
y = X @ w_star + noise

# Adversarial outliers on (1 - alpha) fraction of samples
n_inliers = int(alpha * n)
perm = rng.permutation(n)
inlier_idx = perm[:n_inliers]
outlier_idx = perm[n_inliers:]

y[outlier_idx] = rng.normal(size=len(outlier_idx)) * 10.0

# Fit the list-decodable regressor
ldr = ExpanderLDRRegressor(
    alpha=alpha,
    repetitions=8,
    buckets=200,
    left_degree=3,
    filtering_rounds=4,
    seeds=10,
    blocks=16,
    ridge=1e-3,
    prune_eta=0.2,
    prune_rho=0.1,
    random_state=0,
)
ldr.fit(X, y)

print("List size:", len(ldr.candidates_))
print("Best candidate error:", np.min([
    np.linalg.norm(w - w_star) for w in ldr.candidates_
]))
```

The estimator behaves like a scikit-learn regressor:

- Implements `fit(X, y)`, `predict(X)`, `get_params(deep=True)`, `set_params(**params)`.
- Can be used inside `Pipeline`, `GridSearchCV`, etc.
- Stores all candidates in `candidates_` and a default chosen model in `coef_` / `intercept_`.

---

## High-level algorithm

The implementation closely follows Algorithm 1 from the paper:

1. **Expander sketch construction**
   - For each repetition \(t = 1, \dots, r\), we draw a random left-regular bipartite graph with degree \(d_L\), represented as a mapping from samples to buckets with random signs. This induces bucketed datasets.

2. **Bucket-wise normal equations**
   - For each active bucket \((t, b)\), we form local second-order statistics: \(H_{t,b} = \frac{1}{|I_{t,b}|}\sum_{i \in I_{t,b}} x_i x_i^\top\), \(g_{t,b} = \frac{1}{|I_{t,b}|}\sum_{i \in I_{t,b}} x_i y_i\).

3. **Robust aggregation across buckets**
   - We partition the bucket statistics into blocks, compute block means, and aggregate them using median-of-means or geometric medians to obtain global estimates \(\hat\Sigma\) and \(\hat g\).

4. **Solve sketched normal equations**
   - We obtain a candidate \(\hat\ell\) by solving \((\hat\Sigma + \lambda I)\,\hat\ell = \hat g\).

5. **Spectral filtering**
   - We compute robustly aggregated residual covariances and find the top eigen-direction. If its eigenvalue is too large, we prune buckets that contribute most along this direction and repeat.

6. **List generation**
   - We repeat the whole process over several seeds and cluster the resulting candidates with single-linkage clustering to obtain a final list of regressors.

See `docs/algorithm.md` for detailed pseudo-code and `docs/theory.md` for assumptions and guarantees.

---

## Documentation

- `docs/index.md` – overview and navigation
- `docs/algorithm.md` – algorithmic details and pseudo-code
- `docs/theory.md` – key theorems and assumptions
- `docs/api.md` – public API documentation
- `docs/experiments.md` – how to run & extend experiments

---

## Experiments

Reproduce basic synthetic experiments:

```bash
python -m expander_ldr.experiments --config examples/run_synthetic_experiment.py
```

or run the notebook in `examples/synthetic_demo.ipynb`.

The experiment interface is deliberately simple: it exposes knobs for the contamination level, problem dimension, sample size, and algorithmic parameters, and standard metrics such as recovery error and prediction risk.
