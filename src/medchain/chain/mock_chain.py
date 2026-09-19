"""In-memory ledger mirroring the Solidity contract API exactly.

Purpose: run 200-round experiments in seconds without a node, then flip
`chain.backend: web3` and get identical semantics against Hardhat. Every method
has a counterpart in contracts/FLOrchestrator.sol with the same name and argument
order -- change one, change both; tests/test_chain_parity.py fails if you forget.

Deliberately reproduces the constraints that matter, so results transfer:
  * append-only event log; nothing mutates or deletes history
  * only digests and short CID strings are stored, never weights
  * reputations are integers in basis points (0-10000), because Solidity has no
    floats -- so quantisation error shows up in simulation too, rather than as a
    surprise at deployment
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BPS = 10_000


def to_bps(x: float) -> int:
    return max(0, min(BPS, int(round(x * BPS))))


def from_bps(x: int) -> float:
    return x / BPS


@dataclass
class Event:
    name: str
    block: int
    tx_hash: str
    args: Dict


@dataclass
class Participant:
    address: str
    institution: str
    registered_block: int
    reputation_bps: int = 5_000
    stake: int = 0
    active: bool = True
    total_rewards: int = 0


@dataclass
class Submission:
    round_index: int
    participant: str
    update_hash: str
    cid: str
    num_samples: int
    block: int
    accepted: Optional[bool] = None
    weight_bps: int = 0


@dataclass
class RoundRecord:
    index: int
    global_model_hash: str = ""
    global_model_cid: str = ""
    parent_hash: str = ""
    aggregator: str = ""
    submissions: List[Submission] = field(default_factory=list)
    finalized_block: Optional[int] = None


class MockChain:
    def __init__(self, gas_report: bool = True):
        self.block = 0
        self.events: List[Event] = []
        self.participants: Dict[str, Participant] = {}
        self.rounds: Dict[int, RoundRecord] = {}
        self.token_supply = 0
        self.gas_report = gas_report
        self.gas_used: Dict[str, List[int]] = {}
        self._current_round = -1

    def _mine(self) -> int:
        self.block += 1
        return self.block

    def _tx(self, method: str, payload: str, gas: int) -> str:
        if self.gas_report:
            self.gas_used.setdefault(method, []).append(gas)
        return "0x" + hashlib.sha256(
            f"{method}:{payload}:{self.block}:{time.time_ns()}".encode()).hexdigest()

    def _emit(self, name: str, tx: str, **args) -> None:
        self.events.append(Event(name=name, block=self.block, tx_hash=tx, args=args))

    # -- ParticipantRegistry ----------------------------------------------- #
    def register_participant(self, address: str, institution: str, stake: int = 0) -> str:
        if address in self.participants:
            raise ValueError(f"participant already registered: {address}")
        self._mine()
        tx = self._tx("registerParticipant", address, gas=68_000)
        self.participants[address] = Participant(address=address, institution=institution,
                                                 registered_block=self.block, stake=stake)
        self._emit("ParticipantRegistered", tx, address=address, institution=institution, stake=stake)
        return tx

    def set_reputation(self, address: str, reputation: float) -> str:
        p = self.participants[address]
        self._mine()
        tx = self._tx("updateReputation", address, gas=29_000)
        old = p.reputation_bps
        p.reputation_bps = to_bps(reputation)
        self._emit("ReputationUpdated", tx, address=address, old_bps=old,
                   new_bps=p.reputation_bps, round=self._current_round)
        return tx

    def is_active(self, address: str) -> bool:
        """Same name/shape as Web3Chain.is_active, so FederatedServer never
        reaches into either backend's internals (it used to poke
        `.participants[addr].active` directly, which only MockChain has)."""
        p = self.participants.get(address)
        return bool(p is not None and p.active)

    def slash(self, address: str, amount: int, reason: str) -> str:
        p = self.participants[address]
        self._mine()
        tx = self._tx("slash", address, gas=47_000)
        burned = min(p.stake, amount)
        p.stake -= burned
        p.active = p.stake > 0
        self._emit("ParticipantSlashed", tx, address=address, amount=burned, reason=reason)
        return tx

    # -- FLOrchestrator ---------------------------------------------------- #
    def start_round(self, index: int, aggregator: str, parent_hash: str = "") -> str:
        if index in self.rounds:
            raise ValueError(f"round {index} already started")
        self._mine()
        tx = self._tx("startRound", str(index), gas=52_000)
        self.rounds[index] = RoundRecord(index=index, aggregator=aggregator, parent_hash=parent_hash)
        self._current_round = index
        self._emit("RoundStarted", tx, round=index, aggregator=aggregator, parent_hash=parent_hash)
        return tx

    def submit_update(self, round_index: int, participant: str, update_hash: str,
                      cid: str, num_samples: int) -> str:
        rnd = self.rounds[round_index]
        if rnd.finalized_block is not None:
            raise ValueError(f"round {round_index} already finalized")
        p = self.participants.get(participant)
        if p is None or not p.active:
            raise ValueError(f"participant not registered or inactive: {participant}")
        if any(s.participant == participant for s in rnd.submissions):
            raise ValueError(f"duplicate submission from {participant} in round {round_index}")
        self._mine()
        tx = self._tx("submitUpdate", update_hash, gas=94_000)
        rnd.submissions.append(Submission(round_index, participant, update_hash, cid,
                                          num_samples, self.block))
        self._emit("UpdateSubmitted", tx, round=round_index, participant=participant,
                   update_hash=update_hash, cid=cid, num_samples=num_samples)
        return tx

    def finalize_round(self, round_index: int, global_model_hash: str, global_model_cid: str,
                       accepted: List[str], rejected: List[str],
                       weights_bps: Dict[str, int]) -> str:
        rnd = self.rounds[round_index]
        self._mine()
        tx = self._tx("finalizeRound", global_model_hash, gas=131_000)
        acc = set(accepted)
        for s in rnd.submissions:
            s.accepted = s.participant in acc
            s.weight_bps = weights_bps.get(s.participant, 0)
        rnd.global_model_hash = global_model_hash
        rnd.global_model_cid = global_model_cid
        rnd.finalized_block = self.block
        self._emit("RoundFinalized", tx, round=round_index, model_hash=global_model_hash,
                   cid=global_model_cid, parent_hash=rnd.parent_hash,
                   accepted=sorted(acc), rejected=sorted(rejected))
        return tx

    # -- IncentiveToken ---------------------------------------------------- #
    def mint_rewards(self, round_index: int, shares: Dict[str, float], pool: int) -> str:
        self._mine()
        tx = self._tx("distributeRewards", str(round_index), gas=51_000 + 22_000 * len(shares))
        minted = {}
        for addr, share in shares.items():
            amt = int(round(share * pool))
            if addr in self.participants:
                self.participants[addr].total_rewards += amt
                self.token_supply += amt
                minted[addr] = amt
        self._emit("RewardsDistributed", tx, round=round_index, pool=pool, minted=minted)
        return tx

    # -- views -------------------------------------------------------------- #
    def model_lineage(self) -> List[Dict]:
        """Parent-linked chain of global models: the provenance claim made concrete."""
        return [{"round": i, "hash": r.global_model_hash, "parent": r.parent_hash,
                 "cid": r.global_model_cid, "block": r.finalized_block}
                for i, r in sorted(self.rounds.items()) if r.finalized_block is not None]

    def gas_summary(self) -> Dict[str, Dict[str, float]]:
        """Per-method gas -> the paper's cost table. First thing a systems reviewer
        looks for, and the usual reason blockchain-FL papers are called impractical."""
        out = {}
        for method, uses in self.gas_used.items():
            out[method] = {"calls": len(uses), "mean_gas": sum(uses) / len(uses),
                           "total_gas": sum(uses)}
        out["_TOTAL"] = {"calls": sum(len(v) for v in self.gas_used.values()), "mean_gas": 0.0,
                         "total_gas": sum(sum(v) for v in self.gas_used.values())}
        return out
