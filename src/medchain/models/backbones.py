"""Torch backbones and the StateDict bridge.

The FL core is NumPy-only by design, so this module is the ONLY place torch is
required. Everything crossing into fl/ is Dict[str, np.ndarray].

WHICH LAYERS ARE FEDERATED -- a decision with real consequences:
`get_state` returns trainable parameters AND BatchNorm running statistics by
default. BN statistics leak the client's input distribution and are a documented
FL privacy leak, but excluding them breaks accuracy under non-IID data. Both are
defensible; what is not defensible is failing to say which you did. Set
`include_buffers=False` and switch to GroupNorm (`--norm group`) for the
privacy-preferring configuration -- GroupNorm is also generally the stronger
choice under high label skew.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def build_model(name: str = "resnet50", num_classes: int = 8, pretrained: bool = True,
                norm: str = "batch"):
    import torch.nn as nn
    import torchvision.models as tv

    name = name.lower()
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "vit_b_16":
        m = tv.vit_b_16(weights=tv.ViT_B_16_Weights.DEFAULT if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "smallcnn":
        m = _SmallCNN(num_classes)
    else:
        raise ValueError(f"unknown model '{name}'")

    if norm == "group":
        m = convert_bn_to_gn(m)
    return m


def _SmallCNN(num_classes: int):
    """For MedMNIST-scale dev runs. ImageNet backbones on 28x28 inputs waste
    compute and give misleading accuracies."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes),
    )


def convert_bn_to_gn(module, groups: int = 8):
    """Swap BatchNorm for GroupNorm.

    BN is actively harmful in federated non-IID training: each client's running
    statistics reflect its own skewed distribution, and averaging them across
    clients produces statistics matching no real distribution. GroupNorm has no
    cross-batch state, so this failure mode disappears. Worth an ablation row.
    """
    import torch.nn as nn
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(min(groups, child.num_features),
                                               child.num_features))
        else:
            convert_bn_to_gn(child, groups)
    return module


def get_state(model, include_buffers: bool = True) -> Dict[str, np.ndarray]:
    sd = model.state_dict()
    if not include_buffers:
        trainable = {n for n, _ in model.named_parameters()}
        sd = {k: v for k, v in sd.items() if k in trainable}
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}


def set_state(model, state: Dict[str, np.ndarray], device: str = "cpu") -> None:
    import torch
    current = model.state_dict()
    merged = {k: (torch.from_numpy(np.asarray(state[k])).to(current[k].dtype)
                  if k in state else current[k]) for k in current}
    model.load_state_dict(merged)
    model.to(device)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device(preference: str = "auto") -> str:
    """auto picks the best available backend: CUDA, then Apple Silicon MPS,
    then CPU. Real ISIC-scale training on CPU alone is impractically slow, so
    this is not a cosmetic choice -- it is the difference between a round
    taking seconds and taking hours."""
    import torch
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_torch(model, dataset, device: str = "cpu", batch_size: int = 64,
                   num_workers: int = 0) -> Dict[str, float]:
    """Torch-side counterpart to fl/local_trainer.py's evaluate() -- same three
    metrics (accuracy, loss, macro_f1), same definitions, so numbers from the
    NumPy-tier and torch-tier pipelines are directly comparable in the paper."""
    import torch
    from torch.utils.data import DataLoader

    if len(dataset) == 0:
        return {"accuracy": 0.0, "loss": float("nan"), "macro_f1": 0.0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss, total_n = 0.0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += float(loss_fn(logits, yb).item())
            total_n += yb.size(0)
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(yb.cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    acc = float((pred == true).mean())
    k = int(max(pred.max(initial=0), true.max(initial=0)) + 1)
    f1s = []
    for c in range(k):
        tp = float(((pred == c) & (true == c)).sum())
        fp = float(((pred == c) & (true != c)).sum())
        fn = float(((pred != c) & (true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {"accuracy": acc, "loss": total_loss / max(1, total_n), "macro_f1": float(np.mean(f1s))}


def communication_cost_mb(model, dtype_bytes: int = 4) -> float:
    """Per-client per-round upload in MB.

    Report this. ResNet-50 is ~25.6M parameters ~= 98 MB per client per round;
    at 10 clients and 100 rounds that is ~98 GB of traffic. It is the strongest
    practical argument for on-chain digests plus off-chain payloads, and it makes
    the case for update compression as future work.
    """
    return count_parameters(model) * dtype_bytes / (1024 ** 2)
