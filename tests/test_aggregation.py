"""Each aggregator's DEFINING property, not just 'it runs'."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from medchain.fl.aggregation import (ClientUpdate, coordinate_median, fedavg, krum,
                                     multi_krum, reputation_weighted, trimmed_mean)


def _honest_cohort(n=7, dim=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim).astype(np.float32)
    return [ClientUpdate(f"c{i}", base + rng.normal(scale=0.05, size=dim).astype(np.float32), 100)
           for i in range(n)]


def test_fedavg_is_sample_weighted_mean():
    updates = [ClientUpdate("a", np.array([1.0, 0.0]), 100),
              ClientUpdate("b", np.array([0.0, 1.0]), 300)]
    res = fedavg(updates)
    np.testing.assert_allclose(res.delta, [0.25, 0.75], atol=1e-6)


def test_median_ignores_one_extreme_outlier():
    updates = _honest_cohort(n=5)
    updates[0] = ClientUpdate("attacker", updates[0].delta * 1000, 100)
    res = coordinate_median(updates)
    honest_mean = np.mean([u.delta for u in updates[1:]], axis=0)
    assert np.linalg.norm(res.delta - honest_mean) < np.linalg.norm(updates[0].delta - honest_mean)


def test_trimmed_mean_drops_extremes_at_configured_ratio():
    updates = _honest_cohort(n=10)
    updates[0] = ClientUpdate("lo", updates[0].delta - 100, 100)
    updates[1] = ClientUpdate("hi", updates[1].delta + 100, 100)
    res = trimmed_mean(updates, trim_ratio=0.2)
    assert res.diagnostics["trimmed_per_side"] == 2


def test_krum_selects_a_single_real_client_not_an_interpolation():
    updates = _honest_cohort(n=7)
    res = krum(updates, num_byzantine=1)
    assert len(res.accepted) == 1
    picked = next(u for u in updates if u.client_id == res.accepted[0])
    np.testing.assert_array_equal(res.delta, picked.delta)


def test_multikrum_rejects_a_far_outlier():
    updates = _honest_cohort(n=7)
    updates[0] = ClientUpdate("attacker", updates[0].delta + 50, 100)
    res = multi_krum(updates, num_byzantine=1)
    assert "attacker" not in res.accepted
    assert "attacker" in res.rejected


def test_reputation_weighted_excludes_below_threshold_and_never_returns_empty():
    updates = _honest_cohort(n=4)
    reps = {"c0": 0.9, "c1": 0.9, "c2": 0.05, "c3": 0.9}
    res = reputation_weighted(updates, reputations=reps, trust_threshold=0.15)
    assert "c2" in res.rejected
    assert set(res.accepted) == {"c0", "c1", "c3"}
    assert abs(sum(res.weights.values()) - 1.0) < 1e-6


def test_reputation_weighted_falls_back_when_all_distrusted():
    """Total distrust must not silently zero the model -- that would be a
    trivial denial-of-service on the whole federation."""
    updates = _honest_cohort(n=3)
    reps = {"c0": 0.01, "c1": 0.01, "c2": 0.01}
    res = reputation_weighted(updates, reputations=reps, trust_threshold=0.15)
    assert len(res.accepted) == 3
    assert np.linalg.norm(res.delta) > 0
