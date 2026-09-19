#!/usr/bin/env python3
"""Single experiment: one (dataset, partition, attack, aggregator, DP) cell.

    python experiments/run_simulation.py --config configs/experiments/baseline.yaml

Every run writes results/<name>/{metrics,gas,lineage,reputation,detection}.json.
The sweep scripts (run_attack_sweep.py, run_dp_sweep.py) call this in a loop and
compile the paper tables from those JSON files -- so a single run must be
correct and complete on its own, since it is also the unit everything else
composes from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from medchain.attacks.attacks import select_malicious
from medchain.data.datasets import load_dataset
from medchain.data.partition import earth_mover_skew, partition, partition_report
from medchain.fl.client import NumpyClient
from medchain.fl.local_trainer import MLPConfig, evaluate, init_state
from medchain.fl.reputation import ReputationConfig
from medchain.fl.server import FederatedServer, SimulationConfig
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
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    set_global_seed(raw.get("seed", 0))

    data_cfg = raw.get("data", {"name": "synthetic"})
    ds = load_dataset(data_cfg.pop("name", "synthetic"), **data_cfg)
    print(f"dataset: {ds.name} | train={len(ds.y_train)} test={len(ds.y_test)} "
          f"classes={ds.num_classes} | imbalance={ds.class_balance()['imbalance_ratio']:.2f}x")

    part_cfg = raw.get("partition", {"strategy": "dirichlet", "alpha": 0.5, "num_clients": 10})
    num_clients = part_cfg.pop("num_clients", 10)
    parts = partition(ds.y_train, num_clients, **part_cfg)
    emd = earth_mover_skew(parts, ds.y_train, ds.num_classes)
    print(f"partition: {part_cfg.get('strategy','dirichlet')} | EMD skew={emd:.3f} "
          f"(0=IID, ~2=maximally skewed)")

    attack_cfg = raw.get("attack", {})
    mal_fraction = attack_cfg.get("malicious_fraction", 0.0)
    malicious = select_malicious(sorted(parts.keys()), mal_fraction, seed=raw.get("seed", 0))
    print(f"attack: {attack_cfg.get('name','none')} | malicious={malicious}")

    mlp_cfg = MLPConfig(input_dim=ds.input_dim, num_classes=ds.num_classes,
                        **raw.get("model", {}))
    clients = []
    for i, (cid, idx) in enumerate(sorted(parts.items())):
        clients.append(NumpyClient(
            cid, ds.X_train[idx], ds.y_train[idx], mlp_cfg,
            is_malicious=(cid in malicious),
            data_attack=attack_cfg.get("name") if attack_cfg.get("name") in
                        ("label_flip", "backdoor") else None,
            attack_params=attack_cfg.get("params", {}),
            seed=client_seed(raw.get("seed", 0), cid)))

    sim_cfg = build_config(raw)
    sim_cfg.malicious = malicious
    sim_cfg.attack_name = attack_cfg.get("name", "none")
    sim_cfg.attack_params = attack_cfg.get("params", {})
    sim_cfg.attack_start_round = attack_cfg.get("start_round", 0)
    if args.results_dir:
        sim_cfg.results_dir = args.results_dir

    server = FederatedServer(clients, init_state(mlp_cfg), sim_cfg,
                             evaluate_fn=lambda s: evaluate(s, ds.X_test, ds.y_test))
    server.run(verbose=True)
    out = server.save_results()

    (out / "partition_report.json").write_text(
        __import__("json").dumps(partition_report(parts, ds.y_train, ds.num_classes), indent=2))

    if malicious:
        final = server.ledger.detection_metrics(malicious)
        cumulative = server.ledger.cumulative_detection_metrics(malicious)
        rtd = server.ledger.rounds_to_detect(malicious)
        print(f"\ndetection (final-round snapshot):   precision={final['precision']:.3f} "
              f"recall={final['recall']:.3f} f1={final['f1']:.3f}")
        print(f"detection (ever flagged, cumulative): precision={cumulative['precision']:.3f} "
              f"recall={cumulative['recall']:.3f} f1={cumulative['f1']:.3f}")
        print(f"rounds to first detection: {rtd}")
    print(f"final accuracy: {server.metrics[-1].accuracy:.4f} | "
         f"macro-F1: {server.metrics[-1].macro_f1:.4f}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
