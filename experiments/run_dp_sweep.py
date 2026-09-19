#!/usr/bin/env python3
"""Table 4 generator: the headline privacy/defence trade-off.

Sweeps (epsilon, reputation beta) and reports detection F1 and final accuracy at
each point. This is the experiment the paper's abstract should quote: the round
epsilon at which the reputation mechanism's detection collapses, and how raising
beta buys back detection at the cost of slower response to genuinely new attacks.

    python experiments/run_dp_sweep.py --base configs/experiments/dp_privacy_tradeoff.yaml
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
from medchain.privacy.dp import DPConfig, compute_epsilon, noise_multiplier_for_epsilon
from medchain.utils.seed import client_seed, set_global_seed

TARGET_EPSILONS = [1.0, 3.0, 8.0, float("inf")]   # inf = DP disabled, the no-privacy control
BETAS = [0.5, 0.7, 0.85, 0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="results/sweep_dp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.base).read_text())
    set_global_seed(args.seed)
    data_cfg = dict(raw.get("data", {}))
    ds = load_dataset(data_cfg.pop("name", "synthetic"), **data_cfg)
    part_cfg = dict(raw.get("partition", {}))
    n_clients = part_cfg.pop("num_clients", 10)
    parts = partition(ds.y_train, n_clients, **part_cfg)
    num_rounds = raw.get("simulation", {}).get("num_rounds", 30)
    mal_fraction = raw.get("attack", {}).get("malicious_fraction", 0.3)
    malicious = select_malicious(sorted(parts.keys()), mal_fraction, seed=args.seed)
    sample_rate = raw.get("simulation", {}).get("clients_per_round", 1.0)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for target_eps in TARGET_EPSILONS:
        if target_eps == float("inf"):
            dp_cfg = DPConfig(enabled=False)
            actual_eps = float("inf")
        else:
            z = noise_multiplier_for_epsilon(target_eps, num_rounds, sample_rate)
            actual_eps = compute_epsilon(num_rounds, sample_rate, z)
            dp_cfg = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=z)

        for beta in BETAS:
            mlp_cfg = MLPConfig(input_dim=ds.input_dim, num_classes=ds.num_classes,
                                hidden_dim=32, lr=0.1, batch_size=32, local_epochs=2, seed=args.seed)
            clients = [NumpyClient(cid, ds.X_train[idx], ds.y_train[idx], mlp_cfg,
                                   is_malicious=(cid in malicious), seed=client_seed(args.seed, cid))
                       for cid, idx in sorted(parts.items())]
            sim = SimulationConfig(
                num_rounds=num_rounds, aggregator="reputation",
                reputation=ReputationConfig(beta=beta, threshold_mode="relative"),
                dp=dp_cfg, attack_name="sign_flip", malicious=malicious, seed=args.seed,
                results_dir=str(Path(args.out) / f"eps{target_eps}_beta{beta}"))
            srv = FederatedServer(clients, init_state(mlp_cfg), sim,
                                  evaluate_fn=lambda s: evaluate(s, ds.X_test, ds.y_test))
            srv.run(verbose=False)
            det = srv.ledger.detection_metrics(malicious)
            row = {"target_epsilon": target_eps, "actual_epsilon": round(actual_eps, 3),
                  "beta": beta, "final_accuracy": srv.metrics[-1].accuracy,
                  "detect_precision": det["precision"], "detect_recall": det["recall"],
                  "detect_f1": det["f1"]}
            rows.append(row)
            print(f"eps~{target_eps} (actual={actual_eps:.2f}) beta={beta} "
                 f"-> acc={row['final_accuracy']:.3f} detectF1={det['f1']:.3f}")

    csv_path = Path(args.out) / "dp_tradeoff.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
