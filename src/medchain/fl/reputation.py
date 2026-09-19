"""Cross-round reputation: the mechanism the paper hangs on.

WHY THIS EXISTS
Single-round robust rules (Krum, trimmed mean, median) judge an update only by
its geometry within that round's cohort. They are blind to history, so a client
that trains honestly for 20 rounds and defects at round 21 gets exactly one round
of suspicion; and an honest hospital with a rare case mix -- the entire point of
federating -- is punished identically to an attacker, every round, forever.

Reputation separates those cases by integrating behaviour over time. A rare case
mix gives a consistently-offset but STABLE direction; an attack gives an
inconsistent or adversarially-aligned one. The EMA buys that discrimination.

SCORING: per round, score in [0,1] blending
  s_cos  agreement with a Byzantine-robust reference (coordinate-wise MEDIAN of
         the cohort -- not the mean, which an attacker can drag arbitrarily)
  s_mag  magnitude plausibility as log-space deviation from the median norm, so
         10x-too-large and 10x-too-small are penalised equally
then R_i <- clip(beta*R_i + (1-beta)*s_i, 0, 1).

INTERACTION WITH DP -- THE HEADLINE EXPERIMENT
DP noise is isotropic: it drives every cosine toward 0 and compresses the
honest/malicious gap, so privacy directly erodes the signal the defence needs.
Two mitigations live here:
  1. robust_z normalises WITHIN the cohort (median/MAD) rather than against an
     absolute threshold, so a uniform DP-induced drop largely cancels;
  2. higher beta averages noise across rounds, trading detection LATENCY for
     detection ACCURACY.
Sweeping (epsilon, beta) and reporting where detection collapses is the
contribution. See docs/EXPERIMENT_MATRIX.md Table 4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-12


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD standardisation: resistant to a malicious minority, unlike
    mean/std which the attacker being measured also influences."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad < _EPS:
        return np.zeros_like(values)
    return (values - med) / (1.4826 * mad)


@dataclass
class RoundScores:
    round_index: int
    raw_cosine: Dict[str, float] = field(default_factory=dict)
    norm_ratio: Dict[str, float] = field(default_factory=dict)
    score: Dict[str, float] = field(default_factory=dict)
    reputation_after: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReputationConfig:
    beta: float = 0.8              # EMA memory: higher = slower to condemn AND to forgive
    alpha_cos: float = 0.7         # direction vs magnitude weight
    initial: float = 0.5           # cold start for a newly admitted hospital
    trust_threshold: float = 0.15  # absolute floor; the only threshold if mode="absolute"
    threshold_mode: str = "relative"   # "relative" | "absolute"
    relative_k: float = 1.5        # flag below median - k * scaled MAD of reputations
    use_robust_z: bool = True
    z_scale: float = 2.0


class ReputationLedger:
    """Authoritative off-chain reputation state, mirrored on-chain each round by
    ReputationManager.sol so the trail is externally verifiable."""

    def __init__(self, config: Optional[ReputationConfig] = None):
        self.cfg = config or ReputationConfig()
        self._rep: Dict[str, float] = {}
        self.history: List[RoundScores] = []

    def get(self, client_id: str) -> float:
        return self._rep.setdefault(client_id, self.cfg.initial)

    def snapshot(self) -> Dict[str, float]:
        return dict(self._rep)

    def register(self, client_ids: Sequence[str]) -> None:
        for cid in client_ids:
            self._rep.setdefault(cid, self.cfg.initial)

    def current_threshold(self) -> float:
        """Trust cutoff for this round.

        WHY RELATIVE BY DEFAULT: as the global model converges, honest updates
        become small and noise-dominated, so cohort cosine similarity collapses
        toward zero for EVERYONE -- measured here, mean honest cosine fell from
        ~0.83 at round 0 to ~0.16 by round 14. Every reputation drifts down
        together. A fixed cutoff therefore either catches nobody early or
        everybody late, and which one you get depends on when you stopped
        training. It is measuring convergence, not misbehaviour.

        A cutoff placed relative to the cohort's own reputation distribution
        (median - k * scaled MAD) tracks that drift and keeps asking the question
        that actually matters: is this participant an outlier AMONG ITS PEERS
        right now. The absolute `trust_threshold` is retained as a hard floor so
        a uniformly well-behaved cohort is never punished for being tightly
        clustered.

        Note the honest consequence: at high attacker fractions the median itself
        is captured and this cutoff degrades. That bound belongs in the paper --
        see docs/EXPERIMENT_MATRIX.md Table 5.
        """
        if self.cfg.threshold_mode == "absolute" or len(self._rep) < 3:
            return self.cfg.trust_threshold
        vals = np.array(list(self._rep.values()), dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        if mad < _EPS:
            return self.cfg.trust_threshold
        return float(max(self.cfg.trust_threshold, med - self.cfg.relative_k * mad))

    def trusted(self, client_ids: Sequence[str]) -> List[str]:
        thr = self.current_threshold()
        return [c for c in client_ids if self.get(c) >= thr]

    def score_round(self, client_ids: Sequence[str], deltas: Sequence[np.ndarray],
                    round_index: int = 0) -> RoundScores:
        if len(client_ids) != len(deltas):
            raise ValueError("client_ids and deltas must align")
        self.register(client_ids)

        mat = np.stack(deltas, axis=0)
        reference = np.median(mat, axis=0)

        cos = np.array([cosine(d, reference) for d in mat], dtype=np.float64)
        norms = np.array([float(np.linalg.norm(d)) for d in mat], dtype=np.float64)
        med_norm = float(np.median(norms)) or _EPS
        ratio = norms / med_norm

        if self.cfg.use_robust_z and len(cos) >= 3:
            z = np.clip(robust_z(cos) / self.cfg.z_scale, -30.0, 30.0)  # avoid exp overflow on extreme outliers
            s_cos = 1.0 / (1.0 + np.exp(-z))
        else:
            s_cos = (cos + 1.0) / 2.0

        s_mag = np.exp(-np.abs(np.log(np.clip(ratio, _EPS, None))))

        # GEOMETRIC mean, not arithmetic. A weighted SUM has a structural floor of
        # (1 - alpha_cos) * s_mag: a sign-flipping attacker keeps a perfectly
        # plausible norm, so s_mag ~ 1 props its total score above any threshold
        # below 0.3 and it is never detected, no matter how badly its direction
        # disagrees. Empirically that cost us 0.00 detection recall against
        # sign_flip. The geometric mean encodes the correct semantics -- an update
        # must be plausible in BOTH direction AND magnitude -- so either signal
        # alone can veto, and neither can mask the other. This is also what closes
        # the door on inner_product_manipulation, whose whole design is to look
        # normal in magnitude while being adversarial in direction.
        a = self.cfg.alpha_cos
        score = np.clip(np.power(np.clip(s_cos, _EPS, 1.0), a)
                        * np.power(np.clip(s_mag, _EPS, 1.0), 1.0 - a), 0.0, 1.0)

        rs = RoundScores(round_index=round_index)
        for cid, c, r, s in zip(client_ids, cos, ratio, score):
            rs.raw_cosine[cid] = float(c)
            rs.norm_ratio[cid] = float(r)
            rs.score[cid] = float(s)
        return rs

    def update(self, round_scores: RoundScores) -> Dict[str, float]:
        b = self.cfg.beta
        for cid, s in round_scores.score.items():
            prev = self.get(cid)
            self._rep[cid] = float(np.clip(b * prev + (1.0 - b) * s, 0.0, 1.0))
        round_scores.reputation_after = self.snapshot()
        self.history.append(round_scores)
        return self.snapshot()

    def score_and_update(self, client_ids, deltas, round_index: int = 0) -> Dict[str, float]:
        return self.update(self.score_round(client_ids, deltas, round_index))

    def reward_shares(self, client_ids: Sequence[str],
                      num_samples: Dict[str, int]) -> Dict[str, float]:
        """Token split: reputation x data volume, normalised.

        Not reputation alone -- a client with one image would farm the same reward
        as one with ten thousand. Not volume alone -- that is the classic FL
        incentive failure where fabricating samples pays.
        """
        raw = {c: self.get(c) * float(num_samples.get(c, 0)) for c in client_ids}
        total = sum(raw.values())
        if total <= 0:
            n = max(1, len(client_ids))
            return {c: 1.0 / n for c in client_ids}
        return {c: v / total for c, v in raw.items()}

    def detection_metrics(self, malicious: Sequence[str]) -> Dict[str, float]:
        """Precision/recall of the FINAL-ROUND trust threshold as a detector.

        READ THIS BEFORE REPORTING TABLE 2 NUMBERS: this checks only the last
        round's reputation. With a decaying EMA (moderate beta), a client that
        was correctly flagged and excluded for the first 10 rounds -- real,
        active protection of the trained model -- can drift back above threshold
        by round 30 if its current-round behaviour blends in (this happens
        concretely for sign-flip attacks near convergence, when real updates
        shrink toward zero and a sign-flipped near-zero vector is statistically
        indistinguishable from noise). Reporting ONLY this number for such a run
        would understate the mechanism: the model was protected when it mattered,
        even though the final label says "not detected."

        Use `cumulative_detection_metrics` alongside this one, and prefer
        `rounds_to_detect` as the primary Table 3 metric -- it is the only one of
        the three that cannot be fooled by end-of-training signal decay.
        """
        mal = set(malicious)
        thr = self.current_threshold()
        flagged = {c for c in self._rep if self._rep[c] < thr}
        tp, fp, fn = len(flagged & mal), len(flagged - mal), len(mal - flagged)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1,
                "true_positives": tp, "false_positives": fp, "false_negatives": fn}

    def cumulative_detection_metrics(self, malicious: Sequence[str]) -> Dict[str, float]:
        """Precision/recall of 'was flagged below threshold in AT LEAST ONE
        round' -- the complement to detection_metrics' final-snapshot view.

        This answers "did the mechanism ever protect the model from this
        participant," which is what actually matters for accuracy/robustness
        claims. It is more forgiving of a false positive that shows up once and
        clears, so report BOTH numbers rather than picking whichever looks
        better -- the gap between them is itself informative (a wide gap means
        detection is real but not sustained; near-equal numbers mean detection
        is stable once triggered).
        """
        mal = set(malicious)
        ever_flagged: set = set()
        for rs in self.history:
            vals = np.array(list(rs.reputation_after.values()), dtype=np.float64)
            if self.cfg.threshold_mode == "absolute" or vals.size < 3:
                thr = self.cfg.trust_threshold
            else:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                thr = (self.cfg.trust_threshold if mad < _EPS
                       else max(self.cfg.trust_threshold, med - self.cfg.relative_k * mad))
            ever_flagged |= {c for c, r in rs.reputation_after.items() if r < thr}

        tp, fp, fn = len(ever_flagged & mal), len(ever_flagged - mal), len(mal - ever_flagged)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1,
                "true_positives": tp, "false_positives": fp, "false_negatives": fn}

    def rounds_to_detect(self, malicious: Sequence[str]) -> Dict[str, Optional[int]]:
        """Rounds before each attacker fell below threshold. Table 3."""
        out: Dict[str, Optional[int]] = {c: None for c in malicious}
        for rs in self.history:
            vals = np.array(list(rs.reputation_after.values()), dtype=np.float64)
            if self.cfg.threshold_mode == "absolute" or vals.size < 3:
                thr = self.cfg.trust_threshold
            else:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                thr = (self.cfg.trust_threshold if mad < _EPS
                       else max(self.cfg.trust_threshold, med - self.cfg.relative_k * mad))
            for c in malicious:
                if out[c] is None and rs.reputation_after.get(c, 1.0) < thr:
                    out[c] = rs.round_index
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "config": self.cfg.__dict__,
            "reputation": self._rep,
            "history": [{"round": h.round_index, "cosine": h.raw_cosine,
                         "norm_ratio": h.norm_ratio, "score": h.score,
                         "reputation_after": h.reputation_after} for h in self.history],
        }, indent=2))
