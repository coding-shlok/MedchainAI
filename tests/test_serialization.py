"""Round-trip and determinism guarantees the whole audit trail depends on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from medchain.fl.serialization import (apply_delta, clip_to_norm, compute_delta,
                                       flatten, state_hash, unflatten, vector_hash)


def _state():
    rng = np.random.default_rng(0)
    return {"W1": rng.normal(size=(5, 3)).astype(np.float32),
           "b1": rng.normal(size=(3,)).astype(np.float32)}


def test_hash_is_deterministic():
    s = _state()
    assert state_hash(s) == state_hash({k: v.copy() for k, v in s.items()})


def test_hash_is_key_order_independent():
    s = _state()
    reordered = {"b1": s["b1"], "W1": s["W1"]}
    assert state_hash(s) == state_hash(reordered)


def test_hash_changes_with_content():
    s = _state()
    s2 = {k: v.copy() for k, v in s.items()}
    s2["W1"][0, 0] += 1e-3
    assert state_hash(s) != state_hash(s2)


def test_flatten_unflatten_roundtrip():
    s = _state()
    vec, spec = flatten(s)
    back = unflatten(vec, spec)
    for k in s:
        np.testing.assert_array_equal(s[k], back[k])


def test_delta_apply_inverse():
    a, b = _state(), _state()
    d = compute_delta(a, b)
    back = apply_delta(b, d)
    for k in a:
        np.testing.assert_allclose(a[k], back[k], atol=1e-6)


def test_clip_to_norm_shrinks_only_when_needed():
    v = np.array([3.0, 4.0], dtype=np.float32)   # norm = 5
    assert np.allclose(clip_to_norm(v, 10.0), v)              # under budget: untouched
    clipped = clip_to_norm(v, 1.0)
    assert np.isclose(np.linalg.norm(clipped), 1.0, atol=1e-5)


def test_vector_hash_matches_state_hash_semantics():
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert vector_hash(v1) == vector_hash(v2)
    v2[0] += 1e-4
    assert vector_hash(v1) != vector_hash(v2)
