"""Poisoning attacks, split by where they act.

UPDATE-SPACE attacks corrupt the tensor after local training: cheap to run, easy
to detect, they establish the ceiling of your defence.

DATA-SPACE attacks corrupt the client's training set and let honest SGD produce
the poisoned update. The result looks statistically like a legitimate update from
a hospital with an odd case mix -- the hard case, and where cross-round
reputation should beat single-round geometry.

ADAPTIVE attacks assume the attacker knows the defence. Include at least one; a
defence evaluated only against defence-unaware attacks is the single most common
reviewer objection in this area.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def sign_flip(delta: np.ndarray, scale: float = 1.0, **_) -> np.ndarray:
    """Negate the update. Maximally destructive, trivially detectable."""
    return (-scale * delta).astype(np.float32)


def gaussian_noise(delta: np.ndarray, sigma: float = 1.0,
                   rng: Optional[np.random.Generator] = None, **_) -> np.ndarray:
    rng = rng or np.random.default_rng()
    scale = sigma * (float(np.std(delta)) or 1.0)
    return (delta + rng.normal(0.0, scale, size=delta.shape)).astype(np.float32)


def scaling_attack(delta: np.ndarray, boost: float = 10.0, **_) -> np.ndarray:
    """Model replacement / boosting (Bagdasaryan et al.). Norm clipping alone
    handles this -- include it so the clipping baseline is not a strawman."""
    return (boost * delta).astype(np.float32)


def inner_product_manipulation(delta: np.ndarray, epsilon: float = 0.5, **_) -> np.ndarray:
    """Negative inner product with the true gradient at a plausible norm
    (Xie et al.). Defeats magnitude-only defences."""
    return (-epsilon * delta).astype(np.float32)


def label_flip_map(num_classes: int, mode: str = "cyclic",
                   source: int = 0, target: int = 1) -> Dict[int, int]:
    if mode == "cyclic":
        return {c: (c + 1) % num_classes for c in range(num_classes)}
    if mode == "reverse":
        return {c: num_classes - 1 - c for c in range(num_classes)}
    if mode == "targeted":
        return {source: target}
    raise ValueError(f"unknown label-flip mode '{mode}'")


def apply_label_flip(labels: np.ndarray, mapping: Dict[int, int], fraction: float = 1.0,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Flip `fraction` of eligible labels. fraction < 1 is the stealthy regime,
    where most defences quietly fail -- sweep it."""
    rng = rng or np.random.default_rng()
    out = labels.copy()
    eligible = np.where(np.isin(labels, list(mapping.keys())))[0]
    if eligible.size == 0:
        return out
    n = int(round(fraction * eligible.size))
    chosen = rng.choice(eligible, size=n, replace=False) if n < eligible.size else eligible
    for i in chosen:
        out[i] = mapping[int(labels[i])]
    return out


def apply_backdoor_patch(images: np.ndarray, patch_size: int = 4, value: float = 1.0,
                         corner: str = "br") -> np.ndarray:
    """Stamp a trigger patch on NHWC or NCHW images in [0,1].

    Backdoors are the strongest argument for an audit trail: main-task accuracy is
    untouched, so no metric reveals the compromise. Only provenance tells you
    which round and which participant introduced it.
    """
    out = images.copy()
    if out.ndim != 4:
        raise ValueError("expected a 4-D image batch")
    channels_first = out.shape[1] in (1, 3) and out.shape[-1] not in (1, 3)
    h, w = (out.shape[2], out.shape[3]) if channels_first else (out.shape[1], out.shape[2])
    p = min(patch_size, h, w)
    ys = slice(h - p, h) if "b" in corner else slice(0, p)
    xs = slice(w - p, w) if "r" in corner else slice(0, p)
    if channels_first:
        out[:, :, ys, xs] = value
    else:
        out[:, ys, xs, :] = value
    return out


def colluding_average(deltas: Sequence[np.ndarray], drift: float = -0.3, jitter: float = 0.02,
                      rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Sybils agree with each other and drift jointly off the honest direction.

    This is what breaks Krum: Krum picks the update with the most close
    neighbours, and a colluding bloc manufactures exactly that. Reputation should
    also struggle at high attacker fractions -- report the crossover honestly
    rather than capping the sweep below it.
    """
    rng = rng or np.random.default_rng()
    mean = np.mean(np.stack(deltas, axis=0), axis=0)
    target = drift * mean
    sd = jitter * (float(np.std(mean)) or 1.0)
    return [(target + rng.normal(0, sd, size=mean.shape)).astype(np.float32) for _ in deltas]


def reputation_aware_attack(delta: np.ndarray, reference: np.ndarray,
                            budget: float = 0.3, **_) -> np.ndarray:
    """Defence-aware: stay inside the similarity band that keeps reputation high
    while still injecting a malicious component. `budget` is how far the attacker
    will rotate away from consensus.

    Your defence WILL degrade here. Reporting that degradation is worth more to
    reviewers than a table of wins against naive attacks.
    """
    ref = reference / (float(np.linalg.norm(reference)) or 1.0)
    mal = -delta / (float(np.linalg.norm(delta)) or 1.0)
    mixed = (1.0 - budget) * ref + budget * mal
    mixed /= (float(np.linalg.norm(mixed)) or 1.0)
    return (mixed * float(np.linalg.norm(delta))).astype(np.float32)


UPDATE_ATTACKS = {
    "sign_flip": sign_flip, "gaussian_noise": gaussian_noise,
    "scaling": scaling_attack, "inner_product": inner_product_manipulation,
}


@dataclass
class AttackConfig:
    name: str = "none"
    malicious_fraction: float = 0.0
    start_round: int = 0        # >0 gives the delayed-defection scenario
    params: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


def select_malicious(client_ids: Sequence[str], fraction: float, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    n = int(round(fraction * len(client_ids)))
    if n == 0:
        return []
    return sorted(rng.choice(np.array(client_ids), size=n, replace=False).tolist())
