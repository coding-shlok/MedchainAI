"""Independent auditor: replays the chain and checks the training run is honest.

The point of the blockchain layer is that a regulator who trusts NONE of the
hospitals and NOT the aggregator can still verify what happened. So this module
takes only (a) the event log and (b) the content store. It never touches the
orchestrator's in-memory state, and it must keep working if you delete every
Python object except those two -- that constraint is what makes the audit claim
real rather than decorative.

Five checks, mapping to the paper's integrity claims:
  C1 hash integrity      stored artifact re-hashes to the committed digest
                         (catches silent substitution of weights)
  C2 lineage continuity  each round's parent_hash equals the previous round's
                         model hash (catches history rewriting / forked training)
  C3 submission validity every aggregated update was submitted BEFORE the round
                         was finalized (catches the aggregator inserting its own)
  C4 weight conservation accepted weights sum to 1, rejected carry zero
                         (catches silent inclusion of excluded parties)
  C5 reputation replay   recomputing the EMA from logged scores reproduces the
                         logged reputations (catches favouritism in scoring)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np


class ContentStore(Protocol):
    def get_state(self, cid: str) -> Dict[str, np.ndarray]: ...


@dataclass
class Finding:
    check: str
    severity: str            # "error" | "warning" | "info"
    round_index: Optional[int]
    message: str


@dataclass
class AuditReport:
    findings: List[Finding] = field(default_factory=list)
    rounds_checked: int = 0
    artifacts_verified: int = 0

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def add(self, check: str, severity: str, round_index: Optional[int], message: str) -> None:
        self.findings.append(Finding(check, severity, round_index, message))

    def summary(self) -> str:
        errors = sum(f.severity == "error" for f in self.findings)
        warns = sum(f.severity == "warning" for f in self.findings)
        head = "AUDIT PASSED" if self.passed else "AUDIT FAILED"
        lines = [f"{head}  |  rounds={self.rounds_checked}  artifacts={self.artifacts_verified}  "
                 f"errors={errors}  warnings={warns}"]
        for f in self.findings:
            if f.severity != "info":
                where = f"round {f.round_index}" if f.round_index is not None else "global"
                lines.append(f"  [{f.severity.upper()}] {f.check} @ {where}: {f.message}")
        return "\n".join(lines)


class ChainAuditor:
    def __init__(self, chain, store: Optional[ContentStore] = None, hash_fn=None):
        self.chain = chain
        self.store = store
        if hash_fn is None:
            from ..fl.serialization import state_hash as hash_fn  # noqa: N813
        self.hash_fn = hash_fn

    def verify(self, reputation_history: Optional[List[dict]] = None,
               beta: Optional[float] = None) -> AuditReport:
        rep = AuditReport()
        prev_hash = ""
        for idx, rnd in sorted(self.chain.rounds.items()):
            if rnd.finalized_block is None:
                rep.add("C3", "warning", idx, "round never finalized")
                continue
            rep.rounds_checked += 1

            if self.store is not None and rnd.global_model_cid:
                try:
                    recomputed = self.hash_fn(self.store.get_state(rnd.global_model_cid))
                    if recomputed != rnd.global_model_hash:
                        rep.add("C1", "error", idx,
                                f"stored artifact hashes to {recomputed[:14]}... but chain "
                                f"committed {rnd.global_model_hash[:14]}...")
                    else:
                        rep.artifacts_verified += 1
                except KeyError:
                    rep.add("C1", "error", idx,
                            f"artifact {rnd.global_model_cid[:16]}... missing from store")
                except Exception as e:
                    # A corrupted artifact fails to even DESERIALISE (e.g. a flipped byte
                    # breaks the npz container's own CRC) before the hash comparison ever
                    # runs. That's still evidence of tampering -- report it as a finding,
                    # don't let it crash the whole audit. An auditor that stops running
                    # the moment it hits real corruption isn't catching anything.
                    rep.add("C1", "error", idx,
                            f"artifact {rnd.global_model_cid[:16]}... is corrupted/unreadable "
                            f"({type(e).__name__}: {e})")

            if rep.rounds_checked > 1 and rnd.parent_hash != prev_hash:
                rep.add("C2", "error", idx,
                        f"parent_hash {rnd.parent_hash[:14]}... does not match previous model "
                        f"hash {prev_hash[:14]}...")
            prev_hash = rnd.global_model_hash

            for s in rnd.submissions:
                if s.block > rnd.finalized_block:
                    rep.add("C3", "error", idx,
                            f"submission from {s.participant} at block {s.block} post-dates "
                            f"finalization at {rnd.finalized_block}")

            accepted = [s for s in rnd.submissions if s.accepted]
            total_w = sum(s.weight_bps for s in accepted)
            if accepted and abs(total_w - 10_000) > 50:      # 0.5% quantisation slack
                rep.add("C4", "error", idx,
                        f"accepted weights sum to {total_w} bps, expected 10000")
            for s in rnd.submissions:
                if s.accepted is False and s.weight_bps != 0:
                    rep.add("C4", "error", idx,
                            f"rejected participant {s.participant} carries weight {s.weight_bps}")

        if reputation_history and beta is not None:
            self._replay_reputation(rep, reputation_history, beta)
        if rep.rounds_checked == 0:
            rep.add("C0", "warning", None, "no finalized rounds found on chain")
        return rep

    def _replay_reputation(self, rep: AuditReport, history: List[dict], beta: float) -> None:
        running: Dict[str, float] = {}
        for entry in history:
            idx = entry.get("round")
            for cid, score in entry.get("score", {}).items():
                prev = running.get(cid, entry.get("initial", 0.5))
                running[cid] = float(np.clip(beta * prev + (1 - beta) * score, 0.0, 1.0))
            for cid, published in entry.get("reputation_after", {}).items():
                if cid in running and abs(running[cid] - published) > 1e-3:
                    rep.add("C5", "error", idx,
                            f"reputation for {cid} replays to {running[cid]:.4f} but "
                            f"{published:.4f} was published")

    def trace_model(self, round_index: int) -> List[Dict]:
        """Full ancestry of a deployed model: every round and every participant
        whose data influenced it. The artifact you show a regulator when a model
        is found backdoored and the question is 'who, and when'."""
        out = []
        for idx in range(0, round_index + 1):
            rnd = self.chain.rounds.get(idx)
            if rnd is None or rnd.finalized_block is None:
                continue
            out.append({
                "round": idx, "model_hash": rnd.global_model_hash, "cid": rnd.global_model_cid,
                "block": rnd.finalized_block,
                "contributors": [{"participant": s.participant, "update_hash": s.update_hash,
                                  "num_samples": s.num_samples, "weight_bps": s.weight_bps}
                                 for s in rnd.submissions if s.accepted],
                "excluded": [s.participant for s in rnd.submissions if s.accepted is False]})
        return out

    def influence_by_participant(self, up_to_round: int) -> Dict[str, float]:
        """Cumulative aggregation weight per participant: 'how much of the deployed
        model came from hospital H'. Needed for consent revocation, and for the
        machine-unlearning question a reviewer will raise."""
        totals: Dict[str, float] = {}
        n = 0
        for idx in range(0, up_to_round + 1):
            rnd = self.chain.rounds.get(idx)
            if rnd is None or rnd.finalized_block is None:
                continue
            n += 1
            for s in rnd.submissions:
                if s.accepted:
                    totals[s.participant] = totals.get(s.participant, 0.0) + s.weight_bps / 10_000
        return {k: v / max(1, n) for k, v in sorted(totals.items())}
