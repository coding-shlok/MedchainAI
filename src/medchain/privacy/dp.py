"""Client-level differential privacy.

We apply DP at the CLIENT level (clip + noise the whole update once), not
per-example. Two reasons: it matches the FL threat model -- the aggregator is the
adversary and the unit of secrecy is "what did hospital H contribute" -- and
per-example DP-SGD needs per-sample gradients, which would put ResNet-50 out of
reach on a student GPU.

State this in the paper. Reviewers will ask which granularity you used, and
client-level vs example-level epsilon are NOT comparable numbers.

The accountant is a standard RDP -> (eps, delta) conversion for the subsampled
Gaussian mechanism: adequate for reporting, but cross-check against Opacus for a
camera-ready submission and say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..fl.serialization import clip_to_norm


@dataclass
class DPConfig:
    enabled: bool = False
    clip_norm: float = 1.0          # C: sensitivity bound
    noise_multiplier: float = 1.0   # z: sigma = z * C
    delta: float = 1e-5
    seed: Optional[int] = None

    @property
    def sigma(self) -> float:
        return self.noise_multiplier * self.clip_norm


def privatize_update(delta: np.ndarray, cfg: DPConfig,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Clip THEN noise. Reversing the order destroys the guarantee."""
    if not cfg.enabled:
        return delta.astype(np.float32)
    rng = rng or np.random.default_rng(cfg.seed)
    clipped = clip_to_norm(delta, cfg.clip_norm)
    return (clipped + rng.normal(0.0, cfg.sigma, size=clipped.shape)).astype(np.float32)


def _rdp_gaussian(q: float, z: float, orders: Sequence[float]) -> np.ndarray:
    """RDP of the subsampled Gaussian mechanism (Mironov et al.)."""
    orders = np.asarray(orders, dtype=np.float64)
    if z <= 0:
        return np.full_like(orders, np.inf)
    if q >= 1.0:
        return orders / (2.0 * z ** 2)
    return q ** 2 * orders / (z ** 2) * (1.0 + (q ** 2) * orders / (2.0 * z ** 2))


def compute_epsilon(num_rounds: int, sample_rate: float, noise_multiplier: float,
                    delta: float = 1e-5) -> float:
    """Composed epsilon after `num_rounds` FL rounds."""
    orders = np.concatenate([np.arange(1.1, 11.0, 0.1), np.arange(11.0, 256.0, 1.0)])
    rdp = _rdp_gaussian(sample_rate, noise_multiplier, orders) * num_rounds
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1.0) + np.log((orders - 1.0) / orders)
    eps = np.where(np.isfinite(eps), eps, np.inf)
    return float(np.min(eps))


def noise_multiplier_for_epsilon(target_eps: float, num_rounds: int, sample_rate: float,
                                 delta: float = 1e-5, tol: float = 1e-3) -> float:
    """Binary-search z for a target epsilon, so your sweep is over interpretable
    epsilons (1, 3, 10) rather than arbitrary sigmas."""
    lo, hi = 0.3, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if compute_epsilon(num_rounds, sample_rate, mid, delta) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
