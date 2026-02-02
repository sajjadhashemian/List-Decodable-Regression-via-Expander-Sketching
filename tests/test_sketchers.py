import numpy as np

from expander_ldr.sketchers import FixedBucketSketcher, OneHashSketcher


def test_one_hash_shapes():
    sketcher = OneHashSketcher(n_buckets=5, repetitions=2, random_state=0).fit(
        n_samples=20
    )
    bucket_indices, bucket_signs = sketcher.get_bucket_assignment()
    assert len(bucket_indices) == 2
    assert len(bucket_signs) == 2
    total = sum(len(bucket) for bucket in bucket_indices[0])
    assert total == 20


def test_fixed_bucket_sketcher_returns_input():
    bucket_indices = [[np.array([0, 1]), np.array([2])]]
    bucket_signs = [[np.array([1.0, 1.0]), np.array([1.0])]]
    sketcher = FixedBucketSketcher(bucket_indices=bucket_indices, bucket_signs=bucket_signs)
    sketcher.fit(n_samples=3)
    out_indices, out_signs = sketcher.get_bucket_assignment()
    assert out_indices == bucket_indices
    assert out_signs == bucket_signs
