"""DP mechanics: ordering, epsilon monotonicity, and secure-agg mask cancellation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from medchain.privacy.dp import DPConfig, compute_epsilon, privatize_update
from medchain.privacy.secure_agg import mask_updates, verify_cancellation


def test_privatize_update_disabled_is_identity():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cfg = DPConfig(enabled=False)
    np.testing.assert_array_equal(privatize_update(v, cfg), v)


def test_privatize_update_respects_clip_norm_before_noise():
    v = np.array([100.0, 0.0], dtype=np.float32)   # far over budget
    cfg = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=0.0, seed=0)  # zero noise isolates clipping
    out = privatize_update(v, cfg)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)


def test_epsilon_decreases_as_noise_multiplier_increases():
    e_low_noise = compute_epsilon(num_rounds=10, sample_rate=1.0, noise_multiplier=0.5)
    e_high_noise = compute_epsilon(num_rounds=10, sample_rate=1.0, noise_multiplier=5.0)
    assert e_high_noise < e_low_noise


def test_epsilon_increases_with_more_rounds():
    e_few = compute_epsilon(num_rounds=5, sample_rate=1.0, noise_multiplier=1.0)
    e_many = compute_epsilon(num_rounds=50, sample_rate=1.0, noise_multiplier=1.0)
    assert e_many > e_few


def test_secure_agg_masks_cancel_exactly_when_every_client_present():
    rng = np.random.default_rng(0)
    original = {f"c{i}": rng.normal(size=16).astype(np.float32) for i in range(6)}
    masked = mask_updates(original, base_seed=0, scale=5.0)
    assert verify_cancellation(masked, original)
    # each individual masked value should differ substantially from the original
    for k in original:
        assert not np.allclose(masked[k], original[k], atol=1e-3)
