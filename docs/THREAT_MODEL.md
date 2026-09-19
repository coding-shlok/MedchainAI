# Threat Model

This document fixes what MedChainAI defends against, what it does not, and who
the actors are. A paper that does not state this precisely invites reviewers to
supply their own (usually harsher) threat model in the rebuttal.

## Actors

| Actor | Trusted for | Not trusted for |
|---|---|---|
| Honest hospital | Reporting its own training honestly | Nothing about other hospitals |
| Malicious hospital | Nothing | — |
| Aggregator (orchestrator) | Running the aggregation rule as specified | Not injecting its own updates, not silently reweighting, not colluding with a subset of hospitals |
| Auditor / regulator | Nothing a priori | Verifies everything independently from the public chain + content store |
| Smart contract | Execution matches deployed bytecode (standard blockchain assumption) | — |

The central design commitment: **the auditor trusts no other actor.** Every
integrity claim in `src/medchain/audit/verifier.py` is checked from only the
public event log and the content-addressed store — never from the
orchestrator's in-memory state. If you extend the system, preserve this: any
new claim that requires trusting the orchestrator's memory is not really an
audit claim.

## In scope

- **Byzantine / poisoning clients** — update-space (sign-flip, scaling, inner-product
  manipulation) and data-space (label flip, backdoor). See `src/medchain/attacks/attacks.py`.
- **Colluding minorities (Sybils)** — coordinated drift designed to defeat
  neighbour-based rules like Krum. `attacks.colluding_average`.
- **Defence-aware adversary** — knows the aggregation rule and reputation
  mechanism, optimises within a similarity budget to evade detection.
  `attacks.reputation_aware_attack`. **This is the headline adversary; do not
  relegate it to an appendix.**
- **Dishonest aggregator (partial)** — the chain enforces weight conservation
  (accepted weights sum to 1), submission-before-finalization ordering, and
  hash-linked lineage, so several classes of aggregator misbehaviour are
  either prevented outright or become independently detectable after the fact.
- **History rewriting** — any attempt to alter a past round's committed model
  breaks the parent-hash chain and is caught by `ChainAuditor.verify` (check C2).
- **Free-riding** — reward shares require both reputation and reported sample
  count, so a client that fabricates data or contributes nothing earns
  nothing (see `ReputationLedger.reward_shares`).

## Explicitly out of scope

State these as limitations, not silently — a reviewer who finds an
unstated gap will read it as an oversight rather than a scoping decision.

- **Sybil identity creation is not prevented, only made costly.** Anyone can
  register a new identity by staking. If stake is cheap relative to the
  reward pool, an attacker can mint many low-cost identities. The system's
  only defence here is the stake requirement in `ParticipantRegistry.sol`,
  not detection.
- **Collusion between the aggregator and a subset of hospitals** is not
  detectable by the current audit checks if the aggregator and colluding
  hospitals agree on a consistent, chain-valid but jointly dishonest
  history. Full BFT-style aggregator replacement is future work.
- **Data quality vs. data poisoning** are not perfectly separable. A hospital
  with genuinely unusual case mix and a data-poisoning attacker can produce
  statistically similar updates at low `malicious_fraction` and high
  Dirichlet skew. Report false-positive rate on the attacker-free, high-skew
  cell for exactly this reason (see `docs/EXPERIMENT_MATRIX.md`).
- **Secure aggregation and per-client reputation are in tension.** Full
  cryptographic secure aggregation hides individual updates, which is
  precisely the signal reputation scores. The current `secure_agg.py` module
  is a simplified demonstration of the masking identity, not a claim that
  both properties hold simultaneously in the shipped system. Reconciling
  them (e.g., via zero-knowledge proofs of update well-formedness) is future
  work — say this explicitly in the paper's limitations section.
- **Model extraction / membership inference against the deployed model**
  itself is not addressed; DP here protects the *training-time* update, not
  inference-time queries against the final model.
- **Smart contract security** (reentrancy, upgrade governance, oracle risk)
  is out of scope for a research prototype; `contracts/` is not audited and
  should not be deployed with real funds.
- **Network-level attacks** (eclipse attacks, timing side-channels on
  submission order) are not modeled.

## Privacy unit of analysis

DP is applied at the **client level**: one clip-and-noise operation per
client per round, protecting "did hospital H participate, and roughly what
did it contribute" against an aggregator-side adversary. This is *not*
per-example DP-SGD, and the two are not comparable at the same epsilon value.
State the granularity explicitly whenever epsilon is reported (see
`src/medchain/privacy/dp.py` module docstring).

## What "detection" means

`ReputationLedger.detection_metrics` treats "reputation below the current
trust threshold" as a binary classifier over participants, evaluated against
the ground-truth malicious set that the simulation harness controls. This
is a simulation-only oracle — a real deployment does not have ground truth
and can only report the aggregator-side proxy (reputation trajectory,
rejection counts). Say this plainly in the paper: the precision/recall
numbers are a controlled-experiment measurement of the mechanism, not a
claim about real-world detection rates.
