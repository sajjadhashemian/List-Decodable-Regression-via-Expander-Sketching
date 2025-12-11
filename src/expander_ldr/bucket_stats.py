"""Bucket-wise normal equations and residual covariance estimates."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


class BucketStatistics:
    """Compute bucket-wise statistics for expander-sketched regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self.X_ = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self.y_ = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if self.X_.shape[0] != self.y_.shape[0]:
            raise ValueError("X and y must have matching number of samples")

    def compute_moments(
        self,
        bucket_indices: List[List[np.ndarray]],
        bucket_signs: List[List[np.ndarray]],
        active_buckets: Iterable[Tuple[int, int]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute (H_{t,b}, g_{t,b}) for each active bucket.

        H_{t,b} = (1 / m) sum_{i in bucket} x_i x_i^T
        g_{t,b} = (1 / m) sum_{i in bucket} x_i y_i

        where m is the number of (signed) samples in bucket (t,b).
        Signs are incorporated into x_i and y_i as given in bucket_signs.
        """
        moments: List[Tuple[np.ndarray, np.ndarray]] = []
        for t, b in active_buckets:
            indices = bucket_indices[t][b]
            signs = bucket_signs[t][b]
            if indices.size == 0:
                raise ValueError(f"Bucket {(t, b)} has no samples")
            if indices.shape[0] != signs.shape[0]:
                raise ValueError("Indices and signs must have the same length")

            X_b = self.X_[indices] * signs[:, None]
            y_b = self.y_[indices] * signs
            m = signs.shape[0]
            H = (X_b.T @ X_b) / m
            g = (X_b.T @ y_b) / m
            moments.append((H, g))
        return moments

    def compute_residual_covariances(
        self,
        l_hat: np.ndarray,
        bucket_indices: List[List[np.ndarray]],
        bucket_signs: List[List[np.ndarray]],
        active_buckets: Iterable[Tuple[int, int]],
    ) -> List[np.ndarray]:
        """
        For each active bucket (t,b), compute:
            A_{t,b} : matrix with rows sign * x_i
            b_{t,b} : vector with entries sign * y_i
            r_{t,b} = b_{t,b} - A_{t,b} @ l_hat
            C_{t,b} = A_{t,b}^T diag(r_{t,b}^2) A_{t,b}

        Returns a list of C_{t,b} matrices in the same order as active_buckets.
        """
        l_hat = np.asarray(l_hat, dtype=np.float64)
        covariances: List[np.ndarray] = []
        for t, b in active_buckets:
            indices = bucket_indices[t][b]
            signs = bucket_signs[t][b]
            if indices.size == 0:
                raise ValueError(f"Bucket {(t, b)} has no samples")
            if indices.shape[0] != signs.shape[0]:
                raise ValueError("Indices and signs must have the same length")

            A_b = self.X_[indices] * signs[:, None]
            b_b = self.y_[indices] * signs
            r = b_b - A_b @ l_hat
            weighted_X = A_b * (r**2)[:, None]
            C = A_b.T @ weighted_X
            covariances.append(C)
        return covariances
