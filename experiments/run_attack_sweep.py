#!/usr/bin/env python3
"""Table 1/2/3 generator: aggregator x attack x malicious-fraction grid.

    python experiments/run_attack_sweep.py --base configs/experiments/baseline.yaml \
        --out results/sweep_attacks --rounds 30

For each cell, runs every aggregator against every attack at every malicious
fraction and writes a single CSV. This IS the paper's main results table --
resist the urge to hand-pick cells; run the full grid and report all of it,
including the ones where the reputation aggregator loses.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from medchain.attacks.attacks import select_malicious
from medchain.data.datasets import load_dataset
from medchain.data.partition import partition
from medchain.fl.client import NumpyClient
from medchain.fl.local_trainer import MLPConfig, evaluate, init_state
from medchain.fl.reputation import ReputationConfig
from medchain.fl.server import FederatedServer, SimulationConfig
from medchain.privacy.dp import DPConfig
from medchain.utils.seed import client_seed, set_global_seed

AGGREGATORS = ["fedavg", "median", "trimmed_mean", "krum", "multikrum", "norm_clip", "reputation"]
ATTACKS = ["none", "sign_flip", "scaling", "inner_product", "colluding", "reputation_aware"]
MALICIOUS_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]


def run_cell(ds, parts, aggregator, attack_name, mal_fraction, seed, num_rounds, results_dir):
    malicious = select_malicious(sorted(parts.keys()), mal_fraction, seed=seed) \
        if attack_name != "none" else []
    mlp_cfg = MLPConfig(input_dim=ds.input_dim, num_classes=ds.num_classes,
                        hidden_dim=32, lr=0.1, batch_size=32, local_epochs=2, seed=seed)
    clients = [NumpyClient(cid, ds.X_train[idx], ds.y_train[idx], mlp_cfg,
                           is_malicious=(cid in malicious), seed=client_seed(seed, cid))
               for cid, idx in sorted(parts.items())]

    kwargs = {}
    if aggregator in ("krum", "multikrum"):
        kwargs["num_byzantine"] = max(1, len(malicious)) if malicious else 0

    sim = SimulationConfig(
        num_rounds=num_rounds, aggregator=aggregator, aggregator_kwargs=kwargs,
        reputation=ReputationConfig(beta=0.7), dp=DPConfig(enabled=False),
        attack_name=attack_name if attack_name != "none" else "none",
        malicious=malicious, seed=seed,
        results_dir=str(Path(results_dir) / f"{aggregator}_{attack_name}_{mal_fraction}"))
    srv = FederatedServer(clients, init_state(mlp_cfg), sim,
                          evaluate_fn=lambda s: evaluate(s, ds.X_test, ds.y_test))
    srv.run(verbose=False)

    row = {"aggregator": aggregator, "attack": attack_name, "malicious_fraction": mal_fraction,
          "final_accuracy": srv.metrics[-1].accuracy, "final_macro_f1": srv.metrics[-1].macro_f1,
          "mean_fp": sum(m.false_positives for m in srv.metrics) / len(srv.metrics)}
    if malicious:
        # cumulative ("ever flagged") is the primary Table 2 metric: it answers
        # whether the mechanism protected the model at any point, and is not
        # fooled by end-of-training signal decay the way a final-snapshot check
        # is -- see ReputationLedger.detection_metrics' docstring for why.
        det = srv.ledger.cumulative_detection_metrics(malicious)
        row.update({"detect_precision": det["precision"], "detect_recall": det["recall"],
                   "detect_f1": det["f1"]})
    else:
        row.update({"detect_precision": "", "detect_recall": "", "detect_f1": ""})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="results/sweep_attacks")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.base).read_text())
    set_global_seed(args.seed)
    data_cfg = dict(raw.get("data", {"name": "synthetic"}))
    ds = load_dataset(data_cfg.pop("name", "synthetic"), **data_cfg)
    part_cfg = dict(raw.get("partition", {"strategy": "dirichlet", "alpha": 0.5, "num_clients": 10}))
    n_clients = part_cfg.pop("num_clients", 10)
    parts = partition(ds.y_train, n_clients, **part_cfg)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) / "results.csv"
    fieldnames = ["aggregator", "attack", "malicious_fraction", "final_accuracy",
                 "final_macro_f1", "mean_fp", "detect_precision", "detect_recall", "detect_f1"]

    total = len(AGGREGATORS) * len(ATTACKS) * len(MALICIOUS_FRACTIONS)
    done = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for attack in ATTACKS:
            fractions = [0.0] if attack == "none" else MALICIOUS_FRACTIONS
            for frac in fractions:
                for agg in AGGREGATORS:
                    row = run_cell(ds, parts, agg, attack, frac, args.seed, args.rounds, args.out)
                    writer.writerow(row)
                    f.flush()
                    done += 1
                    print(f"[{done}/{total}] {agg:14s} attack={attack:18s} "
                         f"frac={frac} acc={row['final_accuracy']:.3f}")
    print(f"\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
