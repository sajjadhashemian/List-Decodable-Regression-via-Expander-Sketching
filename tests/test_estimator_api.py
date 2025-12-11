import numpy as np
from sklearn.model_selection import GridSearchCV

from expander_ldr.estimator import ExpanderLDRRegressor


def test_estimator_fit_and_predict_roundtrip():
    rng = np.random.default_rng(0)
    n, d = 60, 4
    alpha = 0.3
    X = rng.standard_normal((n, d))
    coef = rng.standard_normal(d)
    noise = 0.05 * rng.standard_normal(n)
    y = X @ coef + noise

    # Inject some outliers
    outlier_idx = rng.choice(n, size=int((1 - alpha) * n), replace=False)
    y[outlier_idx] += 5.0 * rng.standard_normal(len(outlier_idx))

    model = ExpanderLDRRegressor(
        alpha=alpha,
        seeds=3,
        repetitions=3,
        buckets=30,
        left_degree=2,
        filtering_rounds=2,
        blocks=4,
        random_state=0,
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert model.candidates_.ndim == 2
    assert model.candidates_.shape[1] == d
    assert model.coef_.shape == (d,)
    assert preds.shape == (n,)


def test_sklearn_compatible():
    rng = np.random.default_rng(1)
    n, d = 40, 3
    X = rng.standard_normal((n, d))
    coef = rng.standard_normal(d)
    y = X @ coef + 0.01 * rng.standard_normal(n)

    est = ExpanderLDRRegressor(alpha=0.3, seeds=2, buckets=20, left_degree=2)
    param_grid = {"repetitions": [2, 3]}
    grid = GridSearchCV(est, param_grid, cv=2)
    grid.fit(X, y)

    assert isinstance(grid.best_estimator_, ExpanderLDRRegressor)
