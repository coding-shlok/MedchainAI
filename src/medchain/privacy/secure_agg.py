"""Pairwise-masking secure aggregation (Bonawitz et al., simplified).

Each pair (i,j) derives a shared mask; i adds it and j subtracts it, so masks
cancel in the sum while any individual update is hidden.

WHAT IS SIMPLIFIED, and you must say so in the paper: no Shamir secret sharing
for dropout recovery, no key agreement -- seeds are dealt by the harness. This
demonstrates the aggregation identity, not a deployable protocol. If one client
drops mid-round, the sum does not cancel.

A REAL TENSION TO ACKNOWLEDGE: full secure aggregation hides individual updates,
which is exactly what the reputation mechanism needs to see. Secure aggregation
and per-client reputation are in conflict; reconciling them (e.g. ZK proofs of
update well-formedness) is future work. Do not claim both without qualification.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import numpy as np


def pair_seed(i: str, j: str, base_seed: int = 0) -> int:
    a, b = sorted([i, j])
    digest = hashlib.sha256(f"{a}|{b}|{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_masks(client_ids: Sequence[str], dim: int, base_seed: int = 0,
                   scale: float = 1.0) -> Dict[str, np.ndarray]:
    ids = list(client_ids)
    masks = {c: np.zeros(dim, dtype=np.float32) for c in ids}
    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1:]:
            rng = np.random.default_rng(pair_seed(a, b, base_seed))
            m = rng.normal(0.0, scale, size=dim).astype(np.float32)
            masks[a] += m
            masks[b] -= m
    return masks


def mask_updates(updates: Dict[str, np.ndarray], base_seed: int = 0,
                 scale: float = 1.0) -> Dict[str, np.ndarray]:
    ids = sorted(updates.keys())
    dim = len(next(iter(updates.values())))
    masks = generate_masks(ids, dim, base_seed, scale)
    return {c: (updates[c] + masks[c]).astype(np.float32) for c in ids}


def verify_cancellation(masked: Dict[str, np.ndarray], original: Dict[str, np.ndarray],
                        atol: float = 1e-3) -> bool:
    got = np.sum(np.stack(list(masked.values())), axis=0)
    want = np.sum(np.stack([original[k] for k in masked]), axis=0)
    return bool(np.allclose(got, want, atol=atol))
