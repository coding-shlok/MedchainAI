#!/usr/bin/env python3
"""Centralized (non-federated) training on the full ISIC 2019 dataset.

Purpose: a DIAGNOSTIC, not a paper result. It answers "what accuracy ceiling
does this backbone+data combination actually have, with no federation
overhead at all" -- so we know whether pushing the FEDERATED run further
(more rounds, bigger backbone) is worth the hours, or whether the ceiling
itself needs a different backbone/more data/a coarser task. Compare its
number to results/isic2019/metrics.json's federated final accuracy: that gap
IS the real, reportable cost of federation on this task.

    PYTHONPATH=src python3 experiments/train_centralized.py \\
        --backbone resnet18 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from medchain.data.datasets import isic2019_torch_datasets
from medchain.models.backbones import build_model, evaluate_torch, pick_device
from medchain.utils.seed import set_global_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Dataset_large")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="results/centralized_ceiling")
    args = ap.parse_args()

    import math
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    set_global_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    train_ds, val_ds, train_labels, num_classes = isic2019_torch_datasets(
        root=args.root, image_size=args.image_size, seed=args.seed)
    train_labels = np.asarray(train_labels)
    print(f"dataset: isic2019 | train={len(train_ds)} val={len(val_ds)} classes={num_classes}")

    model = build_model(args.backbone, num_classes, pretrained=True).to(device)

    counts = np.clip(np.bincount(train_labels, minlength=num_classes).astype(np.float64), 1.0, None)
    class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32,
                                 device=device)
    per_sample_weight = (1.0 / counts)[train_labels]
    sampler = WeightedRandomSampler(torch.as_tensor(per_sample_weight, dtype=torch.double),
                                    num_samples=len(train_labels), replacement=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, args.epochs - 1)))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        ev = evaluate_torch(model, val_ds, device=device, batch_size=args.batch_size,
                            num_workers=args.num_workers)
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | lr {lr:.6f} | acc {ev['accuracy']:.4f} | "
              f"f1 {ev['macro_f1']:.4f} | loss {ev['loss']:.4f} | elapsed {elapsed:.0f}s")
        history.append({"epoch": epoch, "lr": lr, **ev})

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nfinal accuracy: {history[-1]['accuracy']:.4f} | macro-F1: {history[-1]['macro_f1']:.4f}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
