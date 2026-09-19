"""Canonical (de)serialization and hashing of model state.

Everything the chain sees is a digest produced here. The digest MUST be
reproducible by an independent auditor months later, so the encoding is pinned:
keys sorted, float32, C-contiguous, little-endian, with shape folded in so two
different tensors sharing bytes cannot collide.

Changing `_canonical_bytes` invalidates every previously published audit trail,
hence ENCODING_VERSION.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

StateDict = Dict[str, np.ndarray]
LayerSpec = List[Tuple[str, Tuple[int, ...]]]

ENCODING_VERSION = b"medchain-v1"


def _canonical_bytes(state: StateDict) -> bytes:
    chunks: List[bytes] = [ENCODING_VERSION]
    for key in sorted(state.keys()):
        arr = np.ascontiguousarray(state[key], dtype="<f4")
        chunks.append(key.encode("utf-8"))
        chunks.append(str(arr.shape).encode("utf-8"))
        chunks.append(arr.tobytes())
    return b"".join(chunks)


def state_hash(state: StateDict) -> str:
    return "0x" + hashlib.sha256(_canonical_bytes(state)).hexdigest()


def vector_hash(vec: np.ndarray) -> str:
    arr = np.ascontiguousarray(vec, dtype="<f4")
    return "0x" + hashlib.sha256(ENCODING_VERSION + arr.tobytes()).hexdigest()


def flatten(state: StateDict) -> Tuple[np.ndarray, LayerSpec]:
    spec: LayerSpec = []
    parts: List[np.ndarray] = []
    for key in sorted(state.keys()):
        arr = np.asarray(state[key], dtype=np.float32)
        spec.append((key, tuple(arr.shape)))
        parts.append(arr.ravel())
    if not parts:
        return np.zeros(0, dtype=np.float32), spec
    return np.concatenate(parts).astype(np.float32), spec


def unflatten(vec: np.ndarray, spec: LayerSpec) -> StateDict:
    out: StateDict = {}
    offset = 0
    for key, shape in spec:
        size = int(np.prod(shape)) if shape else 1
        out[key] = vec[offset: offset + size].reshape(shape).astype(np.float32)
        offset += size
    if offset != vec.size:
        raise ValueError(f"spec covers {offset} params but vector has {vec.size}")
    return out


def compute_delta(new: StateDict, old: StateDict) -> StateDict:
    if set(new.keys()) != set(old.keys()):
        raise ValueError("state dicts have mismatched keys")
    return {k: (np.asarray(new[k], np.float32) - np.asarray(old[k], np.float32)) for k in new}


def apply_delta(base: StateDict, delta: StateDict, scale: float = 1.0) -> StateDict:
    return {k: (np.asarray(base[k], np.float32) + scale * delta[k]).astype(np.float32) for k in base}


def l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def clip_to_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Norm clipping: DP sensitivity bound, and first-line defence against
    magnitude-inflation attacks."""
    norm = l2_norm(vec)
    if norm <= max_norm or norm == 0.0:
        return vec.astype(np.float32)
    return (vec * (max_norm / norm)).astype(np.float32)
