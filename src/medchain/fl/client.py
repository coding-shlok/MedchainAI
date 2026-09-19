"""Hospital-side clients.

Two interchangeable implementations satisfying the same protocol:
    .client_id -> str
    .num_samples -> int
    .train(global_state) -> StateDict

NumpyClient  dependency-free, for CI and protocol debugging
TorchClient  the real thing (models/backbones.py), imported lazily so the
             package still imports on a machine with no torch installed

Data-space attacks (label flip, backdoor) live HERE, not in server.py, because
they must corrupt the training set and let honest SGD produce the update. That is
what makes them statistically hard to distinguish from a hospital with an unusual
case mix -- the whole point of the hard-case evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..attacks.attacks import apply_backdoor_patch, apply_label_flip, label_flip_map
from .local_trainer import MLPConfig, StateDict, train_local


@dataclass
class TorchConfig:
    model_name: str = "resnet18"
    num_classes: int = 8
    pretrained: bool = True
    optimizer: str = "sgd"          # "sgd" or "adam" -- adam is the standard choice for
                                     # fine-tuning a pretrained backbone at low LR
    lr: float = 0.01
    momentum: float = 0.9           # ignored by adam
    weight_decay: float = 1e-4
    batch_size: int = 32
    local_epochs: int = 1
    num_workers: int = 0
    grad_clip: float = 5.0
    balanced_loss: bool = False     # inverse-frequency CrossEntropyLoss weighting, computed
                                     # from EACH CLIENT'S OWN local label counts -- a real
                                     # hospital knows its own case-mix, so this needs no
                                     # cross-client information sharing. Without it, a class
                                     # with 5x fewer images is 5x underweighted in the
                                     # gradient and macro-F1 suffers even when accuracy looks
                                     # fine, because accuracy rewards guessing the majority
                                     # class.
    weighted_sampler: bool = False  # oversample minority classes so they're actually SEEN
                                     # more often per batch, not just weighted more heavily
                                     # when they are. Stacks with balanced_loss; the two fix
                                     # different halves of the same problem (rare class shows
                                     # up rarely vs. rare class counts for less when it does).
    lr_schedule: str = "none"       # "none" or "cosine" -- cosine anneals this client's LR
                                     # to ~0 over `lr_decay_rounds` FEDERATED ROUNDS (not local
                                     # epochs). A constant LR that plateaus accuracy well before
                                     # training stops is the signature this fixes.
    lr_decay_rounds: int = 1        # set to SimulationConfig.num_rounds when using cosine --
                                     # kept separate rather than read automatically because
                                     # TorchConfig has no reference to the simulation config.


class NumpyClient:
    def __init__(self, client_id: str, X: np.ndarray, y: np.ndarray, cfg: MLPConfig,
                 is_malicious: bool = False, data_attack: Optional[str] = None,
                 attack_params: Optional[Dict] = None, seed: int = 0):
        self.client_id = client_id
        self.cfg = cfg
        self.is_malicious = is_malicious
        self.data_attack = data_attack
        self.attack_params = attack_params or {}
        self.rng = np.random.default_rng(seed)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y).astype(int)
        if is_malicious and data_attack:
            self.X, self.y = self._poison(self.X, self.y)

    @property
    def num_samples(self) -> int:
        return int(self.X.shape[0])

    def _poison(self, X: np.ndarray, y: np.ndarray):
        if self.data_attack == "label_flip":
            mapping = label_flip_map(self.cfg.num_classes,
                                     mode=self.attack_params.get("mode", "cyclic"),
                                     source=int(self.attack_params.get("source", 0)),
                                     target=int(self.attack_params.get("target", 1)))
            return X, apply_label_flip(y, mapping,
                                       float(self.attack_params.get("fraction", 1.0)), self.rng)

        if self.data_attack == "backdoor":
            frac = float(self.attack_params.get("poison_fraction", 0.3))
            target = int(self.attack_params.get("target_class", 0))
            shape = self.attack_params.get("image_shape")
            if shape is None:
                raise ValueError("backdoor attack needs attack_params['image_shape'] (H, W, C)")
            n_poison = max(1, int(frac * X.shape[0]))
            idx = self.rng.choice(X.shape[0], size=n_poison, replace=False)
            imgs = X[idx].reshape(-1, *shape)
            imgs = apply_backdoor_patch(imgs, int(self.attack_params.get("patch_size", 4)))
            Xp, yp = X.copy(), y.copy()
            Xp[idx] = imgs.reshape(n_poison, -1)
            yp[idx] = target
            return Xp, yp

        raise ValueError(f"unknown data attack '{self.data_attack}'")

    def train(self, global_state: StateDict) -> StateDict:
        return train_local(global_state, self.X, self.y, self.cfg, self.rng)


class _RelabeledDataset:
    """Wraps a torch Dataset, substituting `labels[i]` for the i-th item's
    label without touching the underlying image loading. Used for label_flip
    poisoning of real-image datasets, where -- unlike the NumPy client -- we
    can't cheaply materialise and mutate a whole `y` array up front."""

    def __init__(self, base, labels: np.ndarray):
        self.base, self.labels = base, labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        import torch
        x, _ = self.base[i]
        return x, torch.tensor(int(self.labels[i]))


class TorchClient:
    """Real training client. torch is imported inside __init__ so that importing
    medchain on a torch-less machine (CI, the auditor's box) still works."""

    def __init__(self, client_id: str, dataset, cfg, is_malicious: bool = False,
                 data_attack: Optional[str] = None, attack_params: Optional[Dict] = None,
                 labels: Optional[np.ndarray] = None, device: str = "cpu", seed: int = 0):
        import torch  # noqa: F401
        from ..models.backbones import build_model, get_state, set_state

        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.is_malicious = is_malicious
        self.data_attack = data_attack
        self.attack_params = attack_params or {}
        self._get_state, self._set_state = get_state, set_state
        self.model = build_model(cfg.model_name, cfg.num_classes,
                                 pretrained=cfg.pretrained).to(device)
        self._seed = seed
        # Kept even when there's no attack: balanced_loss needs this client's own label
        # counts, and unlike NumpyClient we can't cheaply materialise the whole y array
        # from a lazy image Dataset to compute them on demand.
        self._labels = None if labels is None else np.asarray(labels)
        self.dataset = self._maybe_poison(dataset, labels)
        self._round = 0  # incremented once per train() call -- see lr_schedule above

    def _maybe_poison(self, dataset, labels: Optional[np.ndarray]):
        if not (self.is_malicious and self.data_attack):
            return dataset
        if self.data_attack == "label_flip":
            if labels is None:
                raise ValueError("label_flip on TorchClient needs `labels` (the ground-truth "
                                 "labels for `dataset`, in item order) -- pass the array your "
                                 "partition already computed, it costs nothing extra.")
            mapping = label_flip_map(self.cfg.num_classes,
                                     mode=self.attack_params.get("mode", "cyclic"),
                                     source=int(self.attack_params.get("source", 0)),
                                     target=int(self.attack_params.get("target", 1)))
            rng = np.random.default_rng(self._seed)
            poisoned = apply_label_flip(np.asarray(labels), mapping,
                                        float(self.attack_params.get("fraction", 1.0)), rng)
            return _RelabeledDataset(dataset, poisoned)
        raise NotImplementedError(
            f"data attack '{self.data_attack}' is not implemented for TorchClient. "
            "label_flip is supported; backdoor needs patching real pixels pre-normalisation "
            "and hasn't been wired up -- use NumpyClient/MedMNIST for backdoor experiments, "
            "or implement it in TorchClient._maybe_poison before trusting those numbers.")

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def train(self, global_state: StateDict) -> StateDict:
        import torch
        from torch.utils.data import DataLoader

        torch.manual_seed(self._seed)
        self._set_state(self.model, global_state, self.device)
        self.model.train()

        lr = self.cfg.lr
        if self.cfg.lr_schedule == "cosine":
            import math
            t = min(self._round, self.cfg.lr_decay_rounds)
            lr = self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * t / max(1, self.cfg.lr_decay_rounds)))
        elif self.cfg.lr_schedule != "none":
            raise ValueError(f"unknown lr_schedule '{self.cfg.lr_schedule}', use 'none' or 'cosine'")
        self._round += 1

        if self.cfg.optimizer == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr,
                                   weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=lr,
                                  momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"unknown optimizer '{self.cfg.optimizer}', use 'sgd' or 'adam'")

        class_weights = None
        if self.cfg.balanced_loss:
            if self._labels is None:
                raise ValueError("balanced_loss on TorchClient needs `labels` (this client's "
                                 "ground-truth labels, in item order) -- the runner already "
                                 "computes them during partitioning, pass them through.")
            counts = np.bincount(self._labels, minlength=self.cfg.num_classes).astype(np.float64)
            counts = np.clip(counts, 1.0, None)  # a class this client never saw gets no signal
            w = counts.sum() / (self.cfg.num_classes * counts)  # sklearn's "balanced" formula
            class_weights = torch.tensor(w, dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        sampler = None
        shuffle = True
        if self.cfg.weighted_sampler:
            if self._labels is None:
                raise ValueError("weighted_sampler on TorchClient needs `labels`, same as "
                                 "balanced_loss -- see that error for why.")
            counts = np.bincount(self._labels, minlength=self.cfg.num_classes).astype(np.float64)
            counts = np.clip(counts, 1.0, None)
            per_sample_weight = (1.0 / counts)[self._labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(per_sample_weight, dtype=torch.double),
                num_samples=len(self._labels), replacement=True)
            shuffle = False  # sampler and shuffle are mutually exclusive in DataLoader
        loader = DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=shuffle,
                            sampler=sampler, num_workers=self.cfg.num_workers, drop_last=False)

        for _ in range(self.cfg.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                # Local grad clipping: stabilises non-IID training. Separate from
                # the DP clip in privacy/dp.py, which bounds sensitivity on the
                # whole update -- do not conflate them in the paper.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()

        return self._get_state(self.model)
