from __future__ import annotations

"""Utilities for constructing expander sketches and bucket assignments."""

"""Expander sketch construction

We implement the random left-regular bipartite construction used for expander sketching:
for each repetition t and each sample i, choose d_L distinct neighbors (buckets) uniformly
at random (optionally with replacement, yielding a multigraph), and assign each edge an
independent Rademacher sign.

The public API matches the estimator usage:
    expander = ExpanderSketcher(...)
    expander.fit(n_samples)
    bucket_indices, bucket_signs = expander.get_bucket_assignment()
"""

from typing import List, Tuple, Union

import numpy as np


def _as_rng(random_state: Union[int, np.random.Generator, None]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _sample_distinct_neighbors(
    rng: np.random.Generator,
    n_rows: int,
    n_buckets: int,
    degree: int,
    max_resample_rounds: int = 25,
) -> np.ndarray:
    """Return array of shape (n_rows, degree) with distinct entries per row.

    We use a fast vectorized rejection-resampling strategy that is efficient
    when degree is small (constant) as in the paper.
    """
    if degree > n_buckets:
        raise ValueError(
            f"left_degree={degree} cannot exceed n_buckets={n_buckets} "
            "when sampling distinct neighbors. Use allow_replacement=True."
        )

    buckets = rng.integers(0, n_buckets, size=(n_rows, degree), dtype=np.int64)

    for _ in range(max_resample_rounds):
        sb = np.sort(buckets, axis=1)
        bad = np.any(sb[:, 1:] == sb[:, :-1], axis=1)
        if not bad.any():
            return buckets
        buckets[bad] = rng.integers(
            0, n_buckets, size=(bad.sum(), degree), dtype=np.int64
        )

    # Rare fallback: fix remaining bad rows exactly.
    sb = np.sort(buckets, axis=1)
    bad = np.any(sb[:, 1:] == sb[:, :-1], axis=1)
    for i in np.flatnonzero(bad):
        buckets[i] = rng.choice(n_buckets, size=degree, replace=False)
    return buckets


def _invert_to_buckets(
    n_samples: int,
    n_buckets: int,
    buckets_by_sample: np.ndarray,  # (n_samples, d)
    signs_by_sample: np.ndarray,  # (n_samples, d)
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Invert sample->bucket edges into bucket->samples lists.

    Returns:
        bucket_indices[b] = array of sample indices hashed to bucket b
        bucket_signs[b]   = aligned array of signs (+1/-1) for those edges
    """
    d = buckets_by_sample.shape[1]
    left = np.repeat(np.arange(n_samples, dtype=np.int64), d)
    right = buckets_by_sample.reshape(-1).astype(np.int64, copy=False)
    sgn = signs_by_sample.reshape(-1).astype(np.float64, copy=False)

    order = np.argsort(right, kind="mergesort")
    right_sorted = right[order]
    left_sorted = left[order]
    sgn_sorted = sgn[order]

    counts = np.bincount(right_sorted, minlength=n_buckets)
    ends = np.cumsum(counts, dtype=np.int64)

    bucket_indices: List[np.ndarray] = []
    bucket_signs: List[np.ndarray] = []
    start = 0
    for b in range(n_buckets):
        end = int(ends[b])
        bucket_indices.append(left_sorted[start:end])
        bucket_signs.append(sgn_sorted[start:end])
        start = end

    return bucket_indices, bucket_signs


class ExpanderSketcher:
    """Generate random left-regular bipartite graphs for expander sketching.

    Parameters
    ----------
    n_buckets : int
        Number of buckets B per repetition (right side).
    repetitions : int
        Number of independent repetitions r.
    left_degree : int
        Left degree d_L: number of buckets each sample contributes to (per repetition).
    random_state : int | np.random.Generator | None
        Seed or Generator for reproducibility.
    allow_replacement : bool, default=False
        If False, each sample chooses d_L distinct buckets (simple on the left).
        If True, sample buckets with replacement (multigraph), as allowed by the paper.
    """

    def __init__(
        self,
        n_buckets: int,
        repetitions: int,
        left_degree: int,
        random_state: Union[int, np.random.Generator, None] = None,
        allow_replacement: bool = False,
    ) -> None:
        if n_buckets <= 0:
            raise ValueError("n_buckets must be positive.")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive.")
        if left_degree <= 0:
            raise ValueError("left_degree must be positive.")

        self.n_buckets = int(n_buckets)
        self.repetitions = int(repetitions)
        self.left_degree = int(left_degree)
        self.allow_replacement = bool(allow_replacement)
        self.rng_ = _as_rng(random_state)

    def fit(self, n_samples: int) -> "ExpanderSketcher":
        """Sample the expander sketch structure for n_samples."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.n_samples_ = int(n_samples)

        if (not self.allow_replacement) and (self.left_degree > self.n_buckets):
            raise ValueError(
                f"left_degree={self.left_degree} > n_buckets={self.n_buckets} "
                "requires allow_replacement=True."
            )

        self.bucket_indices_: List[List[np.ndarray]] = []
        self.bucket_signs_: List[List[np.ndarray]] = []

        for _ in range(self.repetitions):
            if self.allow_replacement:
                buckets = self.rng_.integers(
                    0,
                    self.n_buckets,
                    size=(self.n_samples_, self.left_degree),
                    dtype=np.int64,
                )
            else:
                buckets = _sample_distinct_neighbors(
                    rng=self.rng_,
                    n_rows=self.n_samples_,
                    n_buckets=self.n_buckets,
                    degree=self.left_degree,
                )

            # Independent Rademacher signs per edge.
            signs = self.rng_.choice(
                np.array([-1.0, 1.0], dtype=np.float64),
                size=(self.n_samples_, self.left_degree),
                replace=True,
            )

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
        """Return (bucket_indices_, bucket_signs_) as lists over repetitions."""
        if not hasattr(self, "bucket_indices_"):
            raise RuntimeError("Call fit(n_samples) before get_bucket_assignment().")
        return self.bucket_indices_, self.bucket_signs_

    def sanity_check_left_degree(self) -> None:
        """Verify each sample participates in exactly left_degree buckets per repetition."""
        if not hasattr(self, "bucket_indices_"):
            raise RuntimeError("Call fit(n_samples) first.")
        n = self.n_samples_
        d = self.left_degree
        B = self.n_buckets
        for t in range(self.repetitions):
            counts = np.zeros(n, dtype=np.int64)
            for b in range(B):
                idx = self.bucket_indices_[t][b]
                counts[idx] += 1
                if idx.shape[0] != self.bucket_signs_[t][b].shape[0]:
                    raise AssertionError("Mismatch between bucket indices and signs.")
            if not np.all(counts == d):
                raise AssertionError(
                    f"Left degree check failed at repetition {t}: "
                    f"min={counts.min()}, max={counts.max()}, expected={d}."
                )


# expander.sanity_check_left_degree()


# from __future__ import annotations

# from typing import List, Tuple

# import networkx as nx
# import numpy as np


# class ExpanderSketcher:
#     """Generate random left-regular bipartite graphs for expander sketching.

#     Parameters
#     ----------
#     n_buckets : int
#         Number of buckets B per repetition (right side).
#     repetitions : int
#         Number of independent repetitions r.
#     left_degree : int
#         Left degree d_L: number of buckets each sample contributes to.
#     random_state : int | np.random.Generator | None
#         Seed or Generator for reproducibility.
#     regularity : {"regular", "configuration_model"}
#         Strategy for generating right-part degrees when constructing the
#         bipartite expander. The default "regular" enforces uniform degrees
#         on the bucket nodes (distributing any remainder as evenly as
#         possible); "configuration_model" samples a multinomial distribution
#         over buckets while keeping the left side regular.
#     """

#     def __init__(
#         self,
#         n_buckets: int,
#         repetitions: int,
#         left_degree: int,
#         random_state=None,
#         regularity: str = "regular",
#     ) -> None:
#         self.n_buckets = n_buckets
#         self.repetitions = repetitions
#         self.left_degree = left_degree
#         self.random_state = random_state
#         self.regularity = regularity
#         self.rng_: np.random.Generator = (
#             random_state
#             if isinstance(random_state, np.random.Generator)
#             else np.random.default_rng(random_state)
#         )

#     def fit(self, n_samples: int) -> "ExpanderSketcher":
#         """Sample the expander structure for n_samples."""

#         self.n_samples_ = n_samples
#         self.bucket_indices_: List[List[np.ndarray]] = []
#         self.bucket_signs_: List[List[np.ndarray]] = []

#         total_edges = n_samples * self.left_degree
#         if self.regularity == "regular":
#             right_degree, remainder = divmod(total_edges, self.n_buckets)
#             right_degrees = [right_degree] * self.n_buckets
#             if remainder:
#                 extra = self.rng_.choice(self.n_buckets, size=remainder, replace=False)
#                 for bucket in extra:
#                     right_degrees[bucket] += 1
#         elif self.regularity == "configuration_model":
#             right_degrees = self.rng_.multinomial(
#                 total_edges, [1 / self.n_buckets] * self.n_buckets
#             ).tolist()
#         else:
#             raise ValueError(
#                 "regularity must be either 'regular' or 'configuration_model'"
#             )

#         left_degrees = [self.left_degree] * n_samples

#         for _ in range(self.repetitions):
#             graph = nx.bipartite.configuration_model(
#                 left_degrees,
#                 right_degrees,
#                 create_using=nx.MultiGraph,
#                 seed=self.rng_,
#             )

#             buckets: List[List[int]] = [[] for _ in range(self.n_buckets)]
#             signs: List[List[float]] = [[] for _ in range(self.n_buckets)]

#             for i in range(n_samples):
#                 bucket_choices = self.rng_.choice(
#                     self.n_buckets, size=self.left_degree, replace=False
#                 )
#                 sign_choices = self.rng_.choice([-1.0, 1.0], size=self.left_degree)
#                 for bucket, sign in zip(bucket_choices, sign_choices):
#                     buckets[bucket].append(i)
#                     signs[bucket].append(sign)
#             edges = []
#             for u, v, k in sorted(graph.edges(keys=True)):
#                 if u < n_samples:
#                     left_node, right_node = u, v - n_samples
#                 else:
#                     left_node, right_node = v, u - n_samples
#                 edges.append((left_node, right_node))

#             sign_choices = self.rng_.choice([-1.0, 1.0], size=len(edges))
#             for (left_node, right_node), sign in zip(edges, sign_choices):
#                 buckets[right_node].append(left_node)
#                 signs[right_node].append(sign)

#             self.bucket_indices_.append(
#                 [np.array(indices, dtype=int) for indices in buckets]
#             )
#             self.bucket_signs_.append(
#                 [np.array(bucket_signs, dtype=float) for bucket_signs in signs]
#             )

#         return self

#     def get_bucket_assignment(
#         self,
#     ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
#         """Return (bucket_indices_, bucket_signs_)."""

#         return self.bucket_indices_, self.bucket_signs_
