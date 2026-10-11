"""Partitioning a centralised dataset into simulated hospitals.

Pick the partition deliberately -- it is the biggest single lever on your headline
numbers, and reviewers know it. Report alpha alongside every accuracy figure or
the result is unfalsifiable.

  iid              sanity check only; nobody believes an IID medical result
  dirichlet(alpha) the standard. alpha->0 pathological, alpha->inf IID.
                   Report alpha in {0.1, 0.5, 1.0}.
  pathological(k)  each client sees exactly k classes (McMahan et al.)
  quantity_skew    equal label mix, power-law dataset SIZES -- models real
                   hospital-size disparity, and is the case where naive
                   sample-count weighting lets one large site dominate
  natural          honour a real site/patient column when the dataset has one

A NOTE THAT MATTERS FOR YOUR CLAIMS: label skew and malicious behaviour look
similar to a defence. A method that "detects attackers" at alpha=1.0 may detect
nothing but rare case mix at alpha=0.1. Always run the attacker-free/high-skew
cell -- false positives there are the honest cost of your method and belong in
the paper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

ClientPartition = Dict[str, np.ndarray]


def _as_ids(num_clients: int) -> List[str]:
    return [f"hospital_{i:02d}" for i in range(num_clients)]


def iid_partition(labels: np.ndarray, num_clients: int, seed: int = 0) -> ClientPartition:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return {cid: part for cid, part in zip(_as_ids(num_clients), np.array_split(idx, num_clients))}


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float = 0.5,
                        seed: int = 0, min_size: int = 10) -> ClientPartition:
    """Hsu et al. label-distribution skew. Retries until every client clears
    `min_size`: empty clients silently change the effective client count and
    corrupt per-round averages."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).ravel()
    classes = np.unique(labels)
    for _ in range(100):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for b, chunk in zip(buckets, np.split(idx_c, cuts)):
                b.extend(chunk.tolist())
        if min(len(b) for b in buckets) >= min_size:
            break
    else:
        raise RuntimeError(f"could not partition with alpha={alpha}, num_clients={num_clients}, "
                           f"min_size={min_size}. Raise alpha or lower num_clients.")
    return {cid: np.array(sorted(b), dtype=int) for cid, b in zip(_as_ids(num_clients), buckets)}


def pathological_partition(labels: np.ndarray, num_clients: int,
                           classes_per_client: int = 2, seed: int = 0) -> ClientPartition:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).ravel()
    classes = np.unique(labels)
    shards_per_class = max(1, int(np.ceil(num_clients * classes_per_client / len(classes))))
    shards: List[np.ndarray] = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        shards.extend(np.array_split(idx_c, shards_per_class))
    rng.shuffle(shards)
    out: ClientPartition = {}
    for i, cid in enumerate(_as_ids(num_clients)):
        take = shards[i * classes_per_client:(i + 1) * classes_per_client]
        out[cid] = np.array(sorted(np.concatenate(take))) if take else np.array([], dtype=int)
    return out


def quantity_skew_partition(labels: np.ndarray, num_clients: int, power: float = 1.5,
                            seed: int = 0, min_size: int = 10) -> ClientPartition:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    sizes = np.array([1.0 / ((i + 1) ** power) for i in range(num_clients)])
    sizes = sizes / sizes.sum()
    counts = np.maximum((sizes * len(idx)).astype(int), min_size)
    counts[-1] = max(min_size, len(idx) - counts[:-1].sum())
    out, start = {}, 0
    for cid, n in zip(_as_ids(num_clients), counts):
        out[cid] = np.array(sorted(idx[start:start + n]))
        start += n
    return out


def natural_partition(site_ids: Sequence, max_clients: Optional[int] = None) -> ClientPartition:
    """Use a real site/patient column. Preferred when available -- a real split is
    worth more than any synthetic alpha."""
    site_ids = np.asarray(site_ids)
    uniq = list(dict.fromkeys(site_ids.tolist()))
    if max_clients:
        uniq = uniq[:max_clients]
    return {f"site_{str(s)}": np.where(site_ids == s)[0] for s in uniq}


def partition(labels: np.ndarray, num_clients: int, strategy: str = "dirichlet",
              seed: int = 0, **kwargs) -> ClientPartition:
    if strategy == "iid":
        return iid_partition(labels, num_clients, seed)
    if strategy == "dirichlet":
        return dirichlet_partition(labels, num_clients, kwargs.get("alpha", 0.5), seed,
                                   kwargs.get("min_size", 10))
    if strategy == "pathological":
        return pathological_partition(labels, num_clients, kwargs.get("classes_per_client", 2), seed)
    if strategy == "quantity_skew":
        return quantity_skew_partition(labels, num_clients, kwargs.get("power", 1.5), seed)
    raise ValueError(f"unknown partition strategy '{strategy}'")


def partition_report(parts: ClientPartition, labels: np.ndarray,
                     num_classes: Optional[int] = None) -> Dict[str, Dict]:
    """Put this table in the paper; it is what makes alpha meaningful."""
    labels = np.asarray(labels).ravel()
    k = num_classes or int(labels.max()) + 1
    report = {}
    for cid, idx in parts.items():
        counts = np.bincount(labels[idx], minlength=k) if idx.size else np.zeros(k, int)
        p = counts / counts.sum() if counts.sum() else counts
        entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        report[cid] = {"num_samples": int(idx.size), "class_counts": counts.tolist(),
                       "label_entropy": entropy,
                       "max_class_share": float(p.max()) if counts.sum() else 0.0}
    return report


def earth_mover_skew(parts: ClientPartition, labels: np.ndarray,
                     num_classes: Optional[int] = None) -> float:
    """Mean L1 distance between each client's label distribution and the global
    one. One scalar to report per experimental cell; 0 = IID."""
    labels = np.asarray(labels).ravel()
    k = num_classes or int(labels.max()) + 1
    glob = np.bincount(labels, minlength=k) / len(labels)
    dists = []
    for idx in parts.values():
        if idx.size == 0:
            continue
        p = np.bincount(labels[idx], minlength=k) / idx.size
        dists.append(float(np.abs(p - glob).sum()))
    return float(np.mean(dists)) if dists else 0.0
