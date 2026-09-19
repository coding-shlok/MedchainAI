"""A small NumPy MLP so the federated pipeline runs with zero deep-learning deps.

This is NOT the model for the paper -- ResNet-50 on ISIC is (models/backbones.py).
It exists so that the FL loop, chain, reputation and audit can be tested in CI in
seconds; you can debug the protocol without a GPU; and a replicator can
`pip install numpy && make demo` and watch the system work, which is worth a
surprising amount at review time.

It shares the StateDict interface with the torch adapter, so swapping in the real
model changes one config line and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

StateDict = Dict[str, np.ndarray]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.size, k), dtype=np.float32)
    out[np.arange(y.size), y.astype(int)] = 1.0
    return out


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 64
    lr: float = 0.05
    batch_size: int = 32
    local_epochs: int = 1
    weight_decay: float = 0.0
    seed: int = 0


def init_state(cfg: MLPConfig) -> StateDict:
    """He initialisation. Every client MUST start a round from the same state or
    the deltas are not comparable and reputation scores become meaningless."""
    rng = np.random.default_rng(cfg.seed)
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / cfg.input_dim),
                         (cfg.input_dim, cfg.hidden_dim)).astype(np.float32),
        "b1": np.zeros(cfg.hidden_dim, dtype=np.float32),
        "W2": rng.normal(0, np.sqrt(2.0 / cfg.hidden_dim),
                         (cfg.hidden_dim, cfg.num_classes)).astype(np.float32),
        "b2": np.zeros(cfg.num_classes, dtype=np.float32),
    }


def forward(state: StateDict, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = np.maximum(0.0, X @ state["W1"] + state["b1"])
    return _softmax(h @ state["W2"] + state["b2"]), h


def train_local(state: StateDict, X: np.ndarray, y: np.ndarray, cfg: MLPConfig,
                rng: Optional[np.random.Generator] = None) -> StateDict:
    rng = rng or np.random.default_rng(cfg.seed)
    s = {k: v.copy() for k, v in state.items()}
    n, k = X.shape[0], cfg.num_classes
    if n == 0:
        return s
    for _ in range(cfg.local_epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb, yb = X[idx], y[idx]
            m = xb.shape[0]
            probs, h = forward(s, xb)
            dlogits = (probs - _one_hot(yb, k)) / m
            gW2 = h.T @ dlogits + cfg.weight_decay * s["W2"]
            gb2 = dlogits.sum(axis=0)
            dh = (dlogits @ s["W2"].T) * (h > 0)
            gW1 = xb.T @ dh + cfg.weight_decay * s["W1"]
            gb1 = dh.sum(axis=0)
            s["W2"] -= cfg.lr * gW2
            s["b2"] -= cfg.lr * gb2
            s["W1"] -= cfg.lr * gW1
            s["b1"] -= cfg.lr * gb1
    return {k: v.astype(np.float32) for k, v in s.items()}


def evaluate(state: StateDict, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if X.shape[0] == 0:
        return {"accuracy": 0.0, "loss": float("nan"), "macro_f1": 0.0}
    probs, _ = forward(state, X)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    loss = float(-np.log(np.clip(probs[np.arange(y.size), y.astype(int)], 1e-12, None)).mean())
    k = probs.shape[1]
    f1s = []
    for c in range(k):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"accuracy": acc, "loss": loss, "macro_f1": float(np.mean(f1s))}


def backdoor_success_rate(state: StateDict, X_triggered: np.ndarray, target_class: int,
                          true_labels: np.ndarray) -> float:
    """Fraction of triggered inputs pushed to the target class, EXCLUDING samples
    already of that class. Forgetting that exclusion inflates the metric by the
    class prior and is a common bug in backdoor papers."""
    mask = true_labels != target_class
    if mask.sum() == 0:
        return 0.0
    probs, _ = forward(state, X_triggered[mask])
    return float((probs.argmax(axis=1) == target_class).mean())
