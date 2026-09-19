"""Partitioning invariants the reported skew numbers depend on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from medchain.data.partition import (dirichlet_partition, earth_mover_skew, iid_partition,
                                     partition_report, pathological_partition)


def _labels(n=2000, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, n)


def test_iid_partition_covers_every_sample_exactly_once():
    y = _labels()
    parts = iid_partition(y, num_clients=8, seed=0)
    all_idx = np.concatenate(list(parts.values()))
    assert len(all_idx) == len(y)
    assert len(set(all_idx.tolist())) == len(y)


def test_dirichlet_low_alpha_is_more_skewed_than_high_alpha():
    y = _labels()
    low = dirichlet_partition(y, num_clients=10, alpha=0.05, seed=0, min_size=5)
    high = dirichlet_partition(y, num_clients=10, alpha=100.0, seed=0, min_size=5)
    assert earth_mover_skew(low, y, 5) > earth_mover_skew(high, y, 5)


def test_pathological_partition_respects_classes_per_client():
    y = _labels(n=3000, k=6)
    parts = pathological_partition(y, num_clients=6, classes_per_client=2, seed=0)
    for idx in parts.values():
        if idx.size:
            assert len(np.unique(y[idx])) <= 2


def test_partition_report_entropy_is_lower_for_skewed_clients():
    y = _labels(n=3000, k=6)
    parts = pathological_partition(y, num_clients=6, classes_per_client=1, seed=0)
    report = partition_report(parts, y, num_classes=6)
    for cid, stats in report.items():
        if stats["num_samples"] > 0:
            assert stats["label_entropy"] < 0.05   # ~single-class clients: near-zero entropy
