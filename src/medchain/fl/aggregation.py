"""Aggregation rules.

All rules share one signature so the harness swaps them by name:
    aggregate(updates: List[ClientUpdate], **kw) -> AggregationResult

`ClientUpdate.delta` is the FLATTENED float32 update. Rules never see raw data
and never see client identity beyond an opaque id -- mirroring what an on-chain
aggregator can actually observe.

Baselines for the paper's comparison table:
    fedavg        McMahan et al. 2017    (no defence -- lower bound)
    median        Yin et al. 2018
    trimmed_mean  Yin et al. 2018
    krum          Blanchard et al. 2017
    multikrum     Blanchard et al. 2017
    norm_clip     Sun et al. 2019
    reputation    OURS

Do not "improve" the baselines. Reviewers check they match the cited papers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .serialization import clip_to_norm


@dataclass
class ClientUpdate:
    client_id: str
    delta: np.ndarray
    num_samples: int
    update_hash: str = ""
    cid: str = ""


@dataclass
class AggregationResult:
    delta: np.ndarray
    weights: Dict[str, float] = field(default_factory=dict)
    accepted: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _stack(updates: List[ClientUpdate]) -> np.ndarray:
    return np.stack([u.delta for u in updates], axis=0)


def _weighted_mean(mat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        return np.zeros(mat.shape[1], dtype=np.float32)
    return ((mat * weights[:, None]).sum(axis=0) / total).astype(np.float32)


def fedavg(updates: List[ClientUpdate], **_) -> AggregationResult:
    mat = _stack(updates)
    w = np.array([u.num_samples for u in updates], dtype=np.float64)
    return AggregationResult(delta=_weighted_mean(mat, w),
                             weights={u.client_id: float(wi / w.sum()) for u, wi in zip(updates, w)},
                             accepted=[u.client_id for u in updates])


def coordinate_median(updates: List[ClientUpdate], **_) -> AggregationResult:
    return AggregationResult(delta=np.median(_stack(updates), axis=0).astype(np.float32),
                             accepted=[u.client_id for u in updates])


def trimmed_mean(updates: List[ClientUpdate], trim_ratio: float = 0.2, **_) -> AggregationResult:
    mat = _stack(updates)
    n = mat.shape[0]
    k = int(np.floor(trim_ratio * n))
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    ordered = np.sort(mat, axis=0)
    kept = ordered[k: n - k] if k > 0 else ordered
    return AggregationResult(delta=kept.mean(axis=0).astype(np.float32),
                             accepted=[u.client_id for u in updates],
                             diagnostics={"trimmed_per_side": float(k)})


def _krum_scores(mat: np.ndarray, num_byzantine: int) -> np.ndarray:
    """Sum of squared distances to the n-f-2 nearest neighbours."""
    n = mat.shape[0]
    sq = np.sum((mat[:, None, :] - mat[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq, np.inf)
    k = max(1, n - num_byzantine - 2)
    return np.sort(sq, axis=1)[:, :k].sum(axis=1)


def krum(updates: List[ClientUpdate], num_byzantine: int = 0, **_) -> AggregationResult:
    mat = _stack(updates)
    pick = int(np.argmin(_krum_scores(mat, num_byzantine)))
    chosen = updates[pick].client_id
    return AggregationResult(delta=mat[pick].astype(np.float32), weights={chosen: 1.0},
                             accepted=[chosen],
                             rejected=[u.client_id for i, u in enumerate(updates) if i != pick])


def multi_krum(updates: List[ClientUpdate], num_byzantine: int = 0,
               num_select: Optional[int] = None, **_) -> AggregationResult:
    mat = _stack(updates)
    n = mat.shape[0]
    m = num_select if num_select is not None else max(1, n - num_byzantine)
    order = np.argsort(_krum_scores(mat, num_byzantine))[:m]
    sel = {int(i) for i in order}
    w = np.array([updates[i].num_samples for i in order], dtype=np.float64)
    return AggregationResult(delta=_weighted_mean(mat[order], w),
                             weights={updates[i].client_id: float(wi / w.sum()) for i, wi in zip(order, w)},
                             accepted=[updates[i].client_id for i in order],
                             rejected=[u.client_id for i, u in enumerate(updates) if i not in sel])


def norm_clipped_fedavg(updates: List[ClientUpdate], clip_norm: float = 1.0, **_) -> AggregationResult:
    clipped = [ClientUpdate(u.client_id, clip_to_norm(u.delta, clip_norm), u.num_samples) for u in updates]
    res = fedavg(clipped)
    res.diagnostics["clip_norm"] = clip_norm
    return res


def reputation_weighted(updates: List[ClientUpdate], reputations: Dict[str, float],
                        trust_threshold: float = 0.15,
                        clip_norm: Optional[float] = None, **_) -> AggregationResult:
    """Weight each client by (reputation x data quantity).

    Reputation is maintained ACROSS rounds by ReputationLedger and mirrored
    on-chain -- the part single-round rules cannot do. A client that behaves for
    20 rounds and defects at round 21 is caught by its history, not by that
    round's geometry alone.

    Clients below `trust_threshold` are excluded and reported as rejected so the
    orchestrator can slash their stake.
    """
    kept, dropped = [], []
    for u in updates:
        (kept if reputations.get(u.client_id, 1.0) >= trust_threshold else dropped).append(u)
    if not kept:                       # never return a zero update on total distrust
        kept, dropped = updates, []
    if clip_norm is not None:
        kept = [ClientUpdate(u.client_id, clip_to_norm(u.delta, clip_norm), u.num_samples) for u in kept]

    mat = _stack(kept)
    w = np.array([reputations.get(u.client_id, 1.0) * float(u.num_samples) for u in kept],
                 dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return AggregationResult(
        delta=_weighted_mean(mat, w),
        weights={u.client_id: float(wi / w.sum()) for u, wi in zip(kept, w)},
        accepted=[u.client_id for u in kept],
        rejected=[u.client_id for u in dropped],
        diagnostics={"mean_reputation": float(np.mean([reputations.get(u.client_id, 1.0) for u in updates]))})


AGGREGATORS: Dict[str, Callable[..., AggregationResult]] = {
    "fedavg": fedavg, "median": coordinate_median, "trimmed_mean": trimmed_mean,
    "krum": krum, "multikrum": multi_krum, "norm_clip": norm_clipped_fedavg,
    "reputation": reputation_weighted,
}


def get_aggregator(name: str) -> Callable[..., AggregationResult]:
    if name not in AGGREGATORS:
        raise KeyError(f"unknown aggregator '{name}'; available: {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]
