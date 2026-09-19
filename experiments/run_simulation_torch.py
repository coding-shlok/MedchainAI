#!/usr/bin/env python3
"""Torch-tier counterpart to run_simulation.py: real backbones (ResNet/DenseNet/
EfficientNet/ViT/SmallCNN) on real images (ISIC 2019 today; any dataset behind
`*_torch_datasets()` in data/datasets.py tomorrow), through the exact same
FederatedServer -- partition -> local train -> attack -> DP -> on-chain commit
-> reputation -> aggregation -> finalize -> reward.

    PYTHONPATH=src python3 experiments/run_simulation_torch.py \\
        --config configs/experiments/isic2019_smoke.yaml

Kept as a separate script (not a branch in run_simulation.py) so the
zero-dependency NumPy path never imports torch, matching the rest of the repo's
"the protocol runs with nothing but numpy" guarantee -- see local_trainer.py.

WHAT THIS DOES NOT YET DO: backdoor poisoning on real images (label_flip is
supported -- see fl/client.py's TorchClient._maybe_poison for exactly why),
and secure aggregation / DP have not been benchmarked at ResNet-50 parameter
counts. Say so if you report numbers from this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from medchain.attacks.attacks import select_malicious
from medchain.data.datasets import isic2019_torch_datasets, isic_folder_torch_datasets
from medchain.data.partition import earth_mover_skew, partition, partition_report
from medchain.fl.client import TorchClient, TorchConfig
from medchain.fl.reputation import ReputationConfig
from medchain.fl.server import FederatedServer, SimulationConfig
from medchain.models.backbones import build_model, communication_cost_mb, evaluate_torch, \
    get_state, pick_device, set_state
from medchain.privacy.dp import DPConfig
from medchain.utils.seed import client_seed, set_global_seed


def build_config(raw: dict) -> SimulationConfig:
    rep = ReputationConfig(**raw.get("reputation", {}))
    dp = DPConfig(**raw.get("dp", {}))
    sim = raw.get("simulation", {})
    return SimulationConfig(reputation=rep, dp=dp, **sim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default=None, help="override config's results_dir")
    ap.add_argument("--device", default=None, help="override config's device (cpu/cuda/mps)")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    seed = raw.get("seed", 0)
    set_global_seed(seed)

    device = pick_device(args.device or raw.get("device", "auto"))
    print(f"device: {device}")

    data_cfg = dict(raw.get("data", {}))
    dataset_name = data_cfg.pop("name", "isic2019")
    class_names = None
    if dataset_name == "isic2019":
        train_ds, val_ds, train_labels, num_classes = isic2019_torch_datasets(seed=seed, **data_cfg)
    elif dataset_name == "isic_folder":
        train_ds, val_ds, train_labels, num_classes, class_names = \
            isic_folder_torch_datasets(seed=seed, **data_cfg)
    else:
        raise ValueError(f"unknown torch-tier dataset '{dataset_name}'. Known: "
                         f"isic2019, isic_folder. Add a branch here for a new one.")
    train_labels = np.asarray(train_labels)
    print(f"dataset: {dataset_name} | train={len(train_ds)} val={len(val_ds)} classes={num_classes}"
          + (f" {class_names}" if class_names else ""))

    part_cfg = dict(raw.get("partition", {"strategy": "dirichlet", "alpha": 0.5, "num_clients": 5}))
    num_clients = part_cfg.pop("num_clients", 5)
    parts = partition(train_labels, num_clients, **part_cfg)
    emd = earth_mover_skew(parts, train_labels, num_classes)
    print(f"partition: {part_cfg.get('strategy','dirichlet')} | EMD skew={emd:.3f} "
          f"(0=IID, ~2=maximally skewed)")

    attack_cfg = raw.get("attack", {})
    mal_fraction = attack_cfg.get("malicious_fraction", 0.0)
    malicious = select_malicious(sorted(parts.keys()), mal_fraction, seed=seed)
    print(f"attack: {attack_cfg.get('name','none')} | malicious={malicious}")

    torch_cfg = TorchConfig(num_classes=num_classes, **raw.get("model", {}))
    print(f"model: {torch_cfg.model_name} | pretrained={torch_cfg.pretrained}")

    from torch.utils.data import Subset

    clients = []
    for cid, idx in sorted(parts.items()):
        subset = Subset(train_ds, idx.tolist())
        clients.append(TorchClient(
            cid, subset, torch_cfg,
            is_malicious=(cid in malicious),
            data_attack=attack_cfg.get("name") if attack_cfg.get("name") == "label_flip" else None,
            attack_params=attack_cfg.get("params", {}),
            labels=train_labels[idx],
            device=device, seed=client_seed(seed, cid)))

    sim_cfg = build_config(raw)
    sim_cfg.malicious = malicious
    sim_cfg.attack_name = attack_cfg.get("name", "none")
    sim_cfg.attack_params = attack_cfg.get("params", {})
    sim_cfg.attack_start_round = attack_cfg.get("start_round", 0)
    if args.results_dir:
        sim_cfg.results_dir = args.results_dir

    eval_model = build_model(torch_cfg.model_name, num_classes, pretrained=torch_cfg.pretrained,
                             norm=raw.get("model", {}).get("norm", "batch")).to(device)
    print(f"communication cost: {communication_cost_mb(eval_model):.1f} MB/client/round")

    def evaluate_fn(state):
        set_state(eval_model, state, device)
        return evaluate_torch(eval_model, val_ds, device=device,
                              batch_size=torch_cfg.batch_size, num_workers=torch_cfg.num_workers)

    initial_state = get_state(eval_model)
    server = FederatedServer(clients, initial_state, sim_cfg, evaluate_fn=evaluate_fn)
    server.run(verbose=True)
    out = server.save_results()

    (out / "partition_report.json").write_text(
        json.dumps(partition_report(parts, train_labels, num_classes), indent=2))

    if malicious:
        final = server.ledger.detection_metrics(malicious)
        cumulative = server.ledger.cumulative_detection_metrics(malicious)
        print(f"\ndetection (final-round snapshot):   precision={final['precision']:.3f} "
              f"recall={final['recall']:.3f} f1={final['f1']:.3f}")
        print(f"detection (ever flagged, cumulative): precision={cumulative['precision']:.3f} "
              f"recall={cumulative['recall']:.3f} f1={cumulative['f1']:.3f}")
    print(f"final accuracy: {server.metrics[-1].accuracy:.4f} | "
         f"macro-F1: {server.metrics[-1].macro_f1:.4f}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
