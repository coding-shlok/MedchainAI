"""Deterministic seeding.

Every stochastic component (partitioning, client init, attacks, DP noise,
cohort sampling) derives its seed from ONE root seed via `client_seed`, so a
full experiment is reproducible from a single integer, and no two components
that should be independent accidentally share a stream.
"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def client_seed(root_seed: int, client_id: str, salt: str = "") -> int:
    """Deterministic per-client seed, independent of client iteration order."""
    import hashlib
    h = hashlib.sha256(f"{root_seed}|{client_id}|{salt}".encode()).digest()
    return int.from_bytes(h[:4], "big")
