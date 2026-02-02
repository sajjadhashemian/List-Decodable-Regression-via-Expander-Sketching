import numpy as np

from expander_ldr.utils import select_by_median_squared_residual


def test_select_by_median_squared_residual():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, 1.0])
    candidates = np.array([[1.0, 1.0], [0.0, 0.0]])
    idx, chosen, scores = select_by_median_squared_residual(X, y, candidates)
    assert idx == 0
    assert np.allclose(chosen, candidates[0])
    assert scores.shape == (2,)
