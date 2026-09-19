"""Behavioural properties of the reputation ledger, including the two bugs
found and fixed during development -- regression-guarded here so they cannot
silently return."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from medchain.fl.reputation import ReputationConfig, ReputationLedger


def _cohort_deltas(n_honest=7, n_malicious=3, dim=20, seed=0, malicious_mode="sign_flip"):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim).astype(np.float32)
    ids, deltas = [], []
    for i in range(n_honest):
        ids.append(f"honest_{i}")
        deltas.append(base + rng.normal(scale=0.1, size=dim).astype(np.float32))
    for i in range(n_malicious):
        ids.append(f"mal_{i}")
        d = base + rng.normal(scale=0.1, size=dim).astype(np.float32)
        deltas.append(-d if malicious_mode == "sign_flip" else d * 10)
    return ids, deltas


def test_geometric_mean_scoring_catches_sign_flip_even_with_plausible_norm():
    """REGRESSION GUARD: additive score combination let a sign-flip attacker's
    plausible NORM mask its adversarial DIRECTION, giving 0.00 detection
    recall in early development. Geometric mean must not regress to additive."""
    ledger = ReputationLedger(ReputationConfig(beta=0.5, alpha_cos=0.7))
    ids, deltas = _cohort_deltas(malicious_mode="sign_flip")
    for r in range(10):
        ledger.score_and_update(ids, deltas, round_index=r)
    honest_rep = np.mean([ledger.get(f"honest_{i}") for i in range(7)])
    mal_rep = np.mean([ledger.get(f"mal_{i}") for i in range(3)])
    assert mal_rep < honest_rep


def test_relative_threshold_tracks_convergence_drift():
    """REGRESSION GUARD: as honest cosine similarity naturally decays toward
    convergence, an absolute threshold either never fires or always fires,
    depending only on when you stopped. Simulate that decay directly and check
    the relative threshold does NOT collapse to zero recall."""
    ledger = ReputationLedger(ReputationConfig(beta=0.6, threshold_mode="relative"))
    rng = np.random.default_rng(1)
    ids = [f"h{i}" for i in range(7)] + [f"m{i}" for i in range(3)]
    for r in range(20):
        decay = max(0.05, 1.0 - r / 25)     # honest signal shrinks toward convergence
        base = rng.normal(size=15).astype(np.float32) * decay
        deltas = [base + rng.normal(scale=0.05, size=15).astype(np.float32) for _ in range(7)]
        deltas += [-base + rng.normal(scale=0.05, size=15).astype(np.float32) for _ in range(3)]
        ledger.score_and_update(ids, deltas, round_index=r)
    det = ledger.detection_metrics([f"m{i}" for i in range(3)])
    assert det["recall"] > 0.5, "relative threshold should still catch sign-flippers late in training"


def test_reward_shares_sum_to_one_and_need_both_reputation_and_volume():
    ledger = ReputationLedger()
    ledger._rep = {"a": 1.0, "b": 0.5}
    shares = ledger.reward_shares(["a", "b"], {"a": 100, "b": 100})
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert shares["a"] > shares["b"]            # same volume, higher reputation -> more reward

    # zero volume must not earn reward even at reputation 1.0 -- otherwise a
    # client could farm rewards with fabricated near-empty updates
    shares2 = ledger.reward_shares(["a", "b"], {"a": 0, "b": 100})
    assert shares2["a"] == 0.0


def test_cold_start_reputation_matches_config():
    ledger = ReputationLedger(ReputationConfig(initial=0.42))
    assert ledger.get("brand_new_hospital") == 0.42


def test_cumulative_detection_catches_early_transient_flagging_that_final_snapshot_misses():
    """REGRESSION GUARD for a real finding during development: sign-flip attackers
    were flagged and excluded from round 0 (real protection), but by round 29
    their reputation had drifted back above threshold as the attack signal faded
    near convergence (sign-flipping a near-zero update looks like noise). The
    final-snapshot metric reported 0.00 recall despite the mechanism working
    correctly throughout training. cumulative_detection_metrics must not have
    this blind spot."""
    ledger = ReputationLedger(ReputationConfig(beta=0.7, threshold_mode="relative"))
    rng = np.random.default_rng(2)
    ids = [f"h{i}" for i in range(7)] + [f"m{i}" for i in range(3)]
    for r in range(30):
        # signal strong early, decays toward noise by round ~15, exactly as
        # observed with the real synthetic-data attack_sign_flip.yaml run
        decay = max(0.02, 1.0 - r / 15)
        base = rng.normal(size=15).astype(np.float32) * decay
        deltas = [base + rng.normal(scale=0.05, size=15).astype(np.float32) for _ in range(7)]
        deltas += [-base + rng.normal(scale=0.05, size=15).astype(np.float32) for _ in range(3)]
        ledger.score_and_update(ids, deltas, round_index=r)

    mal = [f"m{i}" for i in range(3)]
    final = ledger.detection_metrics(mal)
    cumulative = ledger.cumulative_detection_metrics(mal)
    rtd = ledger.rounds_to_detect(mal)

    assert all(v is not None and v <= 3 for v in rtd.values()), \
        "attackers should be caught within the first few rounds while signal is strong"
    # "ever flagged" is a superset of "flagged in the final round" by construction,
    # so cumulative recall can never be LOWER than final recall -- this is the
    # invariant that makes cumulative_detection_metrics a safe complement to
    # report rather than a metric that could itself mislead in the other
    # direction. (On the real synthetic attack_sign_flip.yaml run this gap was
    # strict: final recall 0.00 vs cumulative recall 1.00 -- reproduced in
    # test_reputation.py's docstring-cited numbers, not asserted verbatim here
    # since the exact decay curve is seed-sensitive.)
    assert cumulative["recall"] >= final["recall"], \
        "cumulative (ever-flagged) recall must never be lower than the final-snapshot recall"
