from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def bucket_contamination_stats(bucket_indices, inlier_mask: np.ndarray) -> Dict[str, np.ndarray | float]:
    n_reps = len(bucket_indices)
    n_buckets = len(bucket_indices[0]) if n_reps else 0
    per_bucket_inlier = []
    per_bucket_outlier = []
    bucket_ids: List[Tuple[int, int]] = []

    for t in range(n_reps):
        for b in range(n_buckets):
            idx = bucket_indices[t][b]
            if idx.size == 0:
                inliers = 0
                outliers = 0
            else:
                mask = inlier_mask[idx]
                inliers = int(mask.sum())
                outliers = int(idx.size - inliers)
            per_bucket_inlier.append(inliers)
            per_bucket_outlier.append(outliers)
            bucket_ids.append((t, b))

    per_bucket_inlier_arr = np.array(per_bucket_inlier, dtype=float)
    per_bucket_outlier_arr = np.array(per_bucket_outlier, dtype=float)
    total = per_bucket_inlier_arr + per_bucket_outlier_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        outlier_fraction = np.where(total > 0, per_bucket_outlier_arr / total, 0.0)

    lightly_contaminated = float(
        np.mean(outlier_fraction <= 0.1) if outlier_fraction.size else 0.0
    )

    inlier_unique_counts = np.zeros(inlier_mask.shape[0], dtype=int)
    for t in range(n_reps):
        for b in range(n_buckets):
            idx = bucket_indices[t][b]
            if idx.size == 0:
                continue
            inlier_idx = idx[inlier_mask[idx]]
            if inlier_idx.size == 1:
                inlier_unique_counts[inlier_idx[0]] += 1

    unique_neighbor_stats = {
        "mean": float(inlier_unique_counts[inlier_mask].mean())
        if inlier_mask.any()
        else 0.0,
        "median": float(np.median(inlier_unique_counts[inlier_mask]))
        if inlier_mask.any()
        else 0.0,
        "max": float(inlier_unique_counts[inlier_mask].max())
        if inlier_mask.any()
        else 0.0,
    }

    return {
        "bucket_ids": np.array(bucket_ids, dtype=object),
        "per_bucket_inlier_count": per_bucket_inlier_arr,
        "per_bucket_outlier_count": per_bucket_outlier_arr,
        "outlier_fraction": outlier_fraction,
        "fraction_lightly_contaminated": lightly_contaminated,
        "unique_neighbor_stats_for_inliers": unique_neighbor_stats,
    }


def pruning_precision_recall(
    pruned_buckets_by_round: List[List[Tuple[int, int]]],
    outlier_fraction: Dict[Tuple[int, int], float],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    precision = []
    recall = []
    total_true = sum(1 for frac in outlier_fraction.values() if frac >= threshold)

    for pruned in pruned_buckets_by_round:
        if not pruned:
            precision.append(0.0)
            recall.append(0.0)
            continue
        true_pos = sum(1 for bucket in pruned if outlier_fraction.get(bucket, 0) >= threshold)
        precision.append(true_pos / len(pruned))
        recall.append(true_pos / total_true if total_true > 0 else 0.0)

    return {
        "precision": np.array(precision, dtype=float),
        "recall": np.array(recall, dtype=float),
    }
