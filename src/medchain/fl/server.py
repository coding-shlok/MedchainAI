"""The federated orchestrator: one round = one on-chain transaction sequence.

Round protocol (mirrors FLOrchestrator.sol):
  1. startRound(index, parent_hash)         link this round to the previous
  2. clients train locally on the global state
  3. attack hook mutates malicious clients' updates
  4. DP hook clips + noises every update
  5. clients publish (update_hash, cid); weights go off-chain
  6. reputation scored from cohort geometry, then updated
  7. aggregator combines under the configured rule
  8. finalizeRound(model_hash, cid, accepted, rejected, weights)
  9. rewards minted from reputation x data volume

ORDERING NOTE THAT AFFECTS YOUR RESULTS: DP noise is added BEFORE reputation
scoring (step 4 before 6). That is the honest ordering -- the aggregator only ever
observes privatised updates, so the defence must work on noisy data. Scoring
before noising yields much better detection numbers and a claim you cannot
deploy. Do not "fix" this to improve the tables.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..attacks.attacks import UPDATE_ATTACKS, colluding_average, reputation_aware_attack
from ..chain.mock_chain import MockChain, to_bps
from ..privacy.dp import DPConfig, privatize_update
from ..storage.local_cas import LocalCAS
from .aggregation import ClientUpdate, get_aggregator
from .reputation import ReputationConfig, ReputationLedger
from .serialization import (StateDict, apply_delta, compute_delta, flatten, state_hash,
                            unflatten, vector_hash)


@dataclass
class RoundMetrics:
    round_index: int
    accuracy: float = 0.0
    macro_f1: float = 0.0
    loss: float = float("nan")
    backdoor_asr: float = 0.0
    mean_reputation_honest: float = 0.0
    mean_reputation_malicious: float = 0.0
    num_rejected: int = 0
    false_positives: int = 0


@dataclass
class SimulationConfig:
    num_rounds: int = 30
    clients_per_round: float = 1.0          # participation rate
    aggregator: str = "reputation"
    aggregator_kwargs: Dict = field(default_factory=dict)
    reputation: ReputationConfig = field(default_factory=ReputationConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack_name: str = "none"
    attack_params: Dict = field(default_factory=dict)
    attack_start_round: int = 0
    malicious: List[str] = field(default_factory=list)
    reward_pool_per_round: int = 1_000
    stake_per_participant: int = 100
    slash_amount: int = 10
    seed: int = 0
    results_dir: str = "results/run"
    chain_backend: str = "mock"             # "mock" or "web3"
    chain_kwargs: Dict = field(default_factory=dict)   # passed to Web3Chain(**kwargs)


class FederatedServer:
    """Client-agnostic: anything exposing .client_id, .num_samples and
    .train(global_state) works, so NumpyClient and TorchClient interchange."""

    def __init__(self, clients: Sequence, initial_state: StateDict, config: SimulationConfig,
                 evaluate_fn: Optional[Callable[[StateDict], Dict[str, float]]] = None,
                 backdoor_fn: Optional[Callable[[StateDict], float]] = None):
        self.clients = list(clients)
        self.state = {k: v.copy() for k, v in initial_state.items()}
        self.cfg = config
        self.evaluate_fn = evaluate_fn
        self.backdoor_fn = backdoor_fn

        # `client_id` (e.g. "hospital_00") is what reputation, aggregation, and
        # every report key on -- it never changes. Only the CHAIN needs real
        # Ethereum addresses to sign transactions, so that translation lives
        # entirely behind self._addr() and nowhere else. MockChain's "address"
        # is just a dict key, so it's the identity map there.
        if config.chain_backend == "web3":
            from ..chain.client import Web3Chain
            self.chain = Web3Chain(**config.chain_kwargs)
            # accounts[0] is the deployer/admin (registers nothing, signs the
            # onlyAdmin calls); the rest are assigned to clients in order.
            pool = self.chain.w3.eth.accounts[1:]
            if len(self.clients) > len(pool):
                raise RuntimeError(
                    f"web3 backend: {len(pool)} funded non-admin accounts available but "
                    f"{len(self.clients)} clients configured. Hardhat's default node gives "
                    f"20 accounts (19 usable here) -- reduce num_clients or start the node "
                    f"with more accounts.")
            self._addr_of = {c.client_id: pool[i] for i, c in enumerate(self.clients)}
        else:
            self.chain = MockChain()
            self._addr_of = {c.client_id: c.client_id for c in self.clients}

        self.store = LocalCAS(Path(config.results_dir) / "cas")
        self.ledger = ReputationLedger(config.reputation)
        self.rng = np.random.default_rng(config.seed)
        self.metrics: List[RoundMetrics] = []
        self.malicious = set(config.malicious)

        for c in self.clients:
            self.chain.register_participant(self._addr(c.client_id), institution=c.client_id,
                                            stake=config.stake_per_participant)
        self.ledger.register([c.client_id for c in self.clients])
        self._parent_hash = ""

    def _addr(self, client_id: str) -> str:
        """The one place client_id -> chain address translation happens."""
        return self._addr_of[client_id]

    def _sample_cohort(self) -> List:
        # An ejected participant (stake fully slashed) can no longer submit, so it
        # must be filtered here rather than blowing up at submit_update time.
        eligible = [c for c in self.clients if self.chain.is_active(self._addr(c.client_id))]
        if len(eligible) < 2:
            raise RuntimeError(
                "fewer than 2 active participants remain: every client has been "
                "slashed out. Raise stake_per_participant, lower slash_amount, or "
                "lower reputation.trust_threshold.")
        rate = self.cfg.clients_per_round
        if rate >= 1.0:
            return eligible
        n = max(2, int(round(rate * len(eligible))))
        idx = self.rng.choice(len(eligible), size=n, replace=False)
        return [eligible[i] for i in idx]

    def _apply_attack(self, cohort: List, deltas: Dict[str, np.ndarray],
                      round_index: int) -> Dict[str, np.ndarray]:
        if self.cfg.attack_name in ("none", "") or round_index < self.cfg.attack_start_round:
            return deltas
        mal = [c.client_id for c in cohort if c.client_id in self.malicious]
        if not mal:
            return deltas
        out = dict(deltas)
        name = self.cfg.attack_name

        if name in UPDATE_ATTACKS:
            fn = UPDATE_ATTACKS[name]
            for cid in mal:
                out[cid] = fn(deltas[cid], rng=self.rng, **self.cfg.attack_params)
        elif name == "colluding":
            poisoned = colluding_average([deltas[c] for c in mal], rng=self.rng,
                                         **self.cfg.attack_params)
            for cid, p in zip(mal, poisoned):
                out[cid] = p
        elif name == "reputation_aware":
            honest = [deltas[c.client_id] for c in cohort if c.client_id not in self.malicious]
            ref = (np.median(np.stack(honest), axis=0) if honest
                   else np.zeros_like(next(iter(deltas.values()))))
            for cid in mal:
                out[cid] = reputation_aware_attack(deltas[cid], ref, **self.cfg.attack_params)
        elif name in ("label_flip", "backdoor"):
            pass          # applied to the client's training data -- see client.py
        else:
            raise ValueError(f"unknown attack '{name}'")
        return out

    def run_round(self, round_index: int) -> RoundMetrics:
        cohort = self._sample_cohort()
        self.chain.start_round(round_index, aggregator="orchestrator", parent_hash=self._parent_hash)

        spec = None
        deltas: Dict[str, np.ndarray] = {}
        samples: Dict[str, int] = {}
        for c in cohort:
            trained = c.train(self.state)
            vec, spec = flatten(compute_delta(trained, self.state))
            deltas[c.client_id] = vec
            samples[c.client_id] = int(c.num_samples)

        deltas = self._apply_attack(cohort, deltas, round_index)

        if self.cfg.dp.enabled:
            deltas = {cid: privatize_update(v, self.cfg.dp, self.rng) for cid, v in deltas.items()}

        updates: List[ClientUpdate] = []
        for cid, vec in deltas.items():
            uhash = vector_hash(vec)
            cas_cid = self.store.put_state(unflatten(vec, spec))
            self.chain.submit_update(round_index, self._addr(cid), uhash, cas_cid, samples[cid])
            updates.append(ClientUpdate(cid, vec, samples[cid], uhash, cas_cid))

        reps = self.ledger.score_and_update([u.client_id for u in updates],
                                            [u.delta for u in updates], round_index)
        for cid in deltas:
            self.chain.set_reputation(self._addr(cid), reps[cid])

        kwargs = dict(self.cfg.aggregator_kwargs)
        if self.cfg.aggregator == "reputation":
            kwargs["reputations"] = reps
            kwargs.setdefault("trust_threshold", self.ledger.current_threshold())
        if self.cfg.aggregator in ("krum", "multikrum"):
            kwargs.setdefault("num_byzantine",
                              len([c for c in cohort if c.client_id in self.malicious]))
        result = get_aggregator(self.cfg.aggregator)(updates, **kwargs)
        self.state = apply_delta(self.state, unflatten(result.delta, spec))

        model_hash = state_hash(self.state)
        model_cid = self.store.put_state(self.state)
        self.chain.finalize_round(round_index, model_hash, model_cid,
                                  [self._addr(c) for c in result.accepted],
                                  [self._addr(c) for c in result.rejected],
                                  {self._addr(k): to_bps(v) for k, v in result.weights.items()})
        self._parent_hash = model_hash

        # Slash on SUSTAINED DISTRUST, never on per-round non-selection.
        # `result.rejected` means different things per rule: for Krum it is the
        # 9-of-10 honest clients the rule simply did not pick this round, and
        # slashing those would eject the entire honest cohort within a few rounds.
        # Only reputation below threshold is an accusation of misbehaviour.
        for cid in deltas:
            if reps[cid] < self.ledger.current_threshold():
                self.chain.slash(self._addr(cid), amount=self.cfg.slash_amount,
                                 reason="reputation below trust threshold")

        shares = self.ledger.reward_shares(result.accepted,
                                           {k: samples.get(k, 0) for k in result.accepted})
        self.chain.mint_rewards(round_index, {self._addr(k): v for k, v in shares.items()},
                                self.cfg.reward_pool_per_round)

        return self._collect_metrics(round_index, result, cohort)

    def _collect_metrics(self, round_index: int, result, cohort) -> RoundMetrics:
        m = RoundMetrics(round_index=round_index)
        if self.evaluate_fn:
            ev = self.evaluate_fn(self.state)
            m.accuracy = ev.get("accuracy", 0.0)
            m.macro_f1 = ev.get("macro_f1", 0.0)
            m.loss = ev.get("loss", float("nan"))
        if self.backdoor_fn:
            m.backdoor_asr = self.backdoor_fn(self.state)

        snap = self.ledger.snapshot()
        hon = [snap[c.client_id] for c in cohort if c.client_id not in self.malicious]
        mal = [snap[c.client_id] for c in cohort if c.client_id in self.malicious]
        m.mean_reputation_honest = float(np.mean(hon)) if hon else 0.0
        m.mean_reputation_malicious = float(np.mean(mal)) if mal else 0.0
        m.num_rejected = len(result.rejected)
        m.false_positives = len([c for c in result.rejected if c not in self.malicious])
        self.metrics.append(m)
        return m

    def run(self, verbose: bool = True) -> List[RoundMetrics]:
        for r in range(self.cfg.num_rounds):
            m = self.run_round(r)
            if verbose:
                print(f"round {r:3d} | acc {m.accuracy:.4f} | f1 {m.macro_f1:.4f} | "
                      f"asr {m.backdoor_asr:.3f} | rep(h) {m.mean_reputation_honest:.3f} | "
                      f"rep(m) {m.mean_reputation_malicious:.3f} | "
                      f"rejected {m.num_rejected} (fp {m.false_positives})")
        return self.metrics

    def save_results(self) -> Path:
        out = Path(self.cfg.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps([m.__dict__ for m in self.metrics], indent=2))
        (out / "gas.json").write_text(json.dumps(self.chain.gas_summary(), indent=2))
        (out / "lineage.json").write_text(json.dumps(self.chain.model_lineage(), indent=2))
        self.ledger.save(out / "reputation.json")
        if self.malicious:
            mal = sorted(self.malicious)
            (out / "detection.json").write_text(json.dumps({
                "final_snapshot_metrics": self.ledger.detection_metrics(mal),
                "cumulative_ever_flagged_metrics": self.ledger.cumulative_detection_metrics(mal),
                "rounds_to_detect": self.ledger.rounds_to_detect(mal)}, indent=2))
        return out
