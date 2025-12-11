"""Utilities for constructing expander sketches and bucket assignments."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class ExpanderSketcher:
    """Generate random left-regular bipartite graphs for expander sketching.

    Parameters
    ----------
    n_buckets : int
        Number of buckets B per repetition (right side).
    repetitions : int
        Number of independent repetitions r.
    left_degree : int
        Left degree d_L: number of buckets each sample contributes to.
    random_state : int | np.random.Generator | None
        Seed or Generator for reproducibility.
    """

    def __init__(
        self,
        n_buckets: int,
        repetitions: int,
        left_degree: int,
        random_state=None,
    ) -> None:
        self.n_buckets = n_buckets
        self.repetitions = repetitions
        self.left_degree = left_degree
        self.random_state = random_state
        self.rng_: np.random.Generator = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def fit(self, n_samples: int) -> "ExpanderSketcher":
        """Sample the expander structure for n_samples."""

        self.n_samples_ = n_samples
        self.bucket_indices_: List[List[np.ndarray]] = []
        self.bucket_signs_: List[List[np.ndarray]] = []

        for _ in range(self.repetitions):
            buckets: List[List[int]] = [[] for _ in range(self.n_buckets)]
            signs: List[List[float]] = [[] for _ in range(self.n_buckets)]

            for i in range(n_samples):
                bucket_choices = self.rng_.choice(
                    self.n_buckets, size=self.left_degree, replace=False
                )
                sign_choices = self.rng_.choice([-1.0, 1.0], size=self.left_degree)
                for bucket, sign in zip(bucket_choices, sign_choices):
                    buckets[bucket].append(i)
                    signs[bucket].append(sign)

            self.bucket_indices_.append(
                [np.array(indices, dtype=int) for indices in buckets]
            )
            self.bucket_signs_.append(
                [np.array(bucket_signs, dtype=float) for bucket_signs in signs]
            )

        return self

    def get_bucket_assignment(
        self,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """Return (bucket_indices_, bucket_signs_)."""

        return self.bucket_indices_, self.bucket_signs_
