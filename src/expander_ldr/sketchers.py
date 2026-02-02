from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple, Union

import numpy as np

from .expander import _as_rng, _invert_to_buckets


class BaseSketcher(Protocol):
    def fit(self, n_samples: int) -> "BaseSketcher":
        ...

    def get_bucket_assignment(
        self,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        ...


@dataclass
class OneHashSketcher:
    n_buckets: int
    repetitions: int
    random_state: Union[int, np.random.Generator, None] = None
    use_signs: bool = True

    def fit(self, n_samples: int) -> "OneHashSketcher":
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive.")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")

        self.n_samples_ = int(n_samples)
        rng = _as_rng(self.random_state)
        self.bucket_indices_: List[List[np.ndarray]] = []
        self.bucket_signs_: List[List[np.ndarray]] = []

        for _ in range(self.repetitions):
            buckets = rng.integers(
                0, self.n_buckets, size=(self.n_samples_, 1), dtype=np.int64
            )
            if self.use_signs:
                signs = rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    size=(self.n_samples_, 1),
                    replace=True,
                )
            else:
                signs = np.ones((self.n_samples_, 1), dtype=np.float64)

            bucket_idx, bucket_sgn = _invert_to_buckets(
                n_samples=self.n_samples_,
                n_buckets=self.n_buckets,
                buckets_by_sample=buckets,
                signs_by_sample=signs,
            )
            self.bucket_indices_.append(bucket_idx)
            self.bucket_signs_.append(bucket_sgn)

        return self

    def get_bucket_assignment(
        self,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        if not hasattr(self, "bucket_indices_"):
            raise RuntimeError("Call fit(n_samples) before get_bucket_assignment().")
        return self.bucket_indices_, self.bucket_signs_


@dataclass
class CountSketchSketcher:
    n_buckets: int
    repetitions: int
    random_state: Union[int, np.random.Generator, None] = None
    use_signs: bool = True

    def fit(self, n_samples: int) -> "CountSketchSketcher":
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive.")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")

        self.n_samples_ = int(n_samples)
        rng = _as_rng(self.random_state)
        self.bucket_indices_: List[List[np.ndarray]] = []
        self.bucket_signs_: List[List[np.ndarray]] = []

        for _ in range(self.repetitions):
            buckets = rng.integers(
                0, self.n_buckets, size=(self.n_samples_, 1), dtype=np.int64
            )
            if self.use_signs:
                signs = rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float64),
                    size=(self.n_samples_, 1),
                    replace=True,
                )
            else:
                signs = np.ones((self.n_samples_, 1), dtype=np.float64)

            bucket_idx, bucket_sgn = _invert_to_buckets(
                n_samples=self.n_samples_,
                n_buckets=self.n_buckets,
                buckets_by_sample=buckets,
                signs_by_sample=signs,
            )
            self.bucket_indices_.append(bucket_idx)
            self.bucket_signs_.append(bucket_sgn)

        return self

    def get_bucket_assignment(
        self,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        if not hasattr(self, "bucket_indices_"):
            raise RuntimeError("Call fit(n_samples) before get_bucket_assignment().")
        return self.bucket_indices_, self.bucket_signs_


@dataclass
class FixedBucketSketcher:
    bucket_indices: List[List[np.ndarray]]
    bucket_signs: List[List[np.ndarray]]

    def fit(self, n_samples: int) -> "FixedBucketSketcher":
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.n_samples_ = int(n_samples)
        return self

    def get_bucket_assignment(
        self,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        return self.bucket_indices, self.bucket_signs
