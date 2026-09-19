# Experiment Matrix

This is the run plan. Each table maps directly to a script and a results file,
so "which experiment produced Table 3" is never ambiguous during writeup.

Run the small/dev-tier version of everything first (`data.name: synthetic` or
`medmnist:*`, num_rounds ~15-30) to catch bugs cheaply, then re-run the same
configs against ISIC with num_rounds ~100-150 for the numbers that go in the
paper. Never hand-edit a results CSV — regenerate it.

---

## Table 1 — Non-IID sensitivity (no attack)

**Question:** how much does label skew alone cost each aggregation rule,
before any adversary is introduced? This is the control every other table is
read against.

Run: `experiments/run_simulation.py` with `attack.name: none`, sweeping
`partition.alpha` over `{100 (~IID), 1.0, 0.5, 0.1}` for each aggregator.

Report: final accuracy, macro-F1, and `partition.py`'s earth-mover skew
number per cell (not just alpha — alpha is a generator parameter, EMD skew is
what the reader can actually compare across datasets).

```bash
for alpha in 100 1.0 0.5 0.1; do
  for agg in fedavg median trimmed_mean krum multikrum reputation; do
    python experiments/run_simulation.py --config configs/experiments/baseline.yaml \
      --results-dir results/table1/${agg}_alpha${alpha}
    # (edit partition.alpha and simulation.aggregator in a copy of the config,
    #  or extend run_simulation.py with --override flags before the full sweep)
  done
done
```

---

## Table 2 — Detection precision/recall/F1 by attack type

**Question:** does the reputation mechanism actually catch each attack class,
and at what false-positive cost against honest clients?

Run: `experiments/run_attack_sweep.py` — this is a single command that
produces the whole grid (aggregator x attack x malicious_fraction).

```bash
python experiments/run_attack_sweep.py \
  --base configs/experiments/baseline.yaml --out results/sweep_attacks --rounds 100
```

Output: `results/sweep_attacks/results.csv`. Columns already match the table
shape — pivot on `attack` (rows) x `aggregator` (columns), report
`detect_precision` / `detect_recall` / `detect_f1` and `final_accuracy`
side by side.

**Which detection metric, and why it matters:** `ReputationLedger` exposes
two. `detection_metrics()` checks only the *final* round's reputation.
`cumulative_detection_metrics()` checks whether a participant was ever
flagged below threshold at any point in training. These can disagree
substantially — confirmed during development with the sign-flip attack on
synthetic data: attackers were flagged and excluded from **round 0**
(real protection, verified via `rounds_to_detect`), but by round 29 their
reputation had drifted back above threshold as the attack's relative signal
faded near convergence (a sign-flipped near-zero update looks like noise).
The final-snapshot metric alone reported 0.00 recall despite the mechanism
working correctly throughout training. `run_attack_sweep.py` reports the
cumulative metric for exactly this reason — report both if space allows,
and always report `rounds_to_detect` (Table 3) alongside either, since it is
the one number immune to this blind spot.

**Report every cell, including the ones where `reputation` loses.** The
`reputation_aware` row against `reputation_aware_attack` at
`malicious_fraction=0.4` is expected to be weak — that is the honest
boundary of the method, not a bug to hide.

**Mandatory control row:** `attack.name: none` at every `malicious_fraction`
value the sweep uses elsewhere (the sweep script does this automatically —
see the `false_positive` column). This is the honest cost of the defence on
a purely honest, merely skewed federation.

---

## Table 3 — Rounds-to-detect (delayed defection)

**Question:** if a hospital behaves honestly for N rounds and then defects,
how many additional rounds does reputation need to catch it? This is the
experiment that differentiates cross-round reputation from single-round
rules like Krum, which cannot express "delayed defection" as a concept at
all — write that framing into the table caption.

Run: `run_simulation.py` with `attack.start_round` set to `{0, 10, 20, 30}`
against a fixed `num_rounds` (e.g. 60), single aggregator = `reputation`.
Read `results/<run>/detection.json`'s `rounds_to_detect` field directly —
`ReputationLedger.rounds_to_detect` computes exactly this.

```yaml
# configs/experiments/delayed_defection_round20.yaml
attack: {name: sign_flip, malicious_fraction: 0.2, start_round: 20}
simulation: {num_rounds: 60, aggregator: reputation, results_dir: results/table3/start20}
```

---

## Table 4 — Privacy/detection trade-off (THE HEADLINE EXPERIMENT)

**Question:** at what DP epsilon does reputation-based detection collapse,
and does raising the EMA memory (`beta`) buy back detection at the cost of
slower response? This is the result the abstract should quote.

Run: `experiments/run_dp_sweep.py` — sweeps target epsilon in
`{1, 3, 8, inf}` x beta in `{0.5, 0.7, 0.85, 0.95}` against a fixed attack.

```bash
python experiments/run_dp_sweep.py --base configs/experiments/dp_privacy_tradeoff.yaml
```

Output: `results/sweep_dp/dp_tradeoff.csv`. Plot detection F1 (y) against
actual epsilon (x, log scale), one line per beta value. The crossover point
— the epsilon below which no beta value recovers useful detection — is the
number to put in the abstract.

State the DP granularity (client-level, not per-example) every time epsilon
is reported here — see `docs/THREAT_MODEL.md`.

---

## Table 5 — Threshold mode ablation (relative vs. absolute)

**Question:** does cohort-relative thresholding actually earn its
complexity over a fixed threshold? This is a real ablation from a bug found
during development (see `tests/test_reputation.py`,
`test_relative_threshold_tracks_convergence_drift`) — the absolute
threshold's detection recall degrades as training converges and honest
cosine similarity naturally decays toward zero.

Run: `run_simulation.py` with `reputation.threshold_mode` in
`{relative, absolute}`, same attack config, sweeping `num_rounds` in
`{20, 60, 120}` to show the drift effect growing with training length.

**Also report the crossover with attacker fraction.** `current_threshold()`'s
docstring notes that at high attacker fractions the cohort median itself
becomes attacker-influenced and the relative threshold degrades — sweep
`malicious_fraction` up to 0.5 and show where this happens. This is the
honest bound of the relative-threshold design; report it rather than
capping the sweep just below it.

---

## Table 6 — On-chain cost (deployability)

**Question:** is this actually affordable to run? First thing a systems
reviewer checks, and the most common reason blockchain-FL papers are
rejected as impractical.

Run against the REAL contracts, not MockChain:

```bash
npm install
make chain-up                          # starts Hardhat node + deploys
# in configs, set chain.backend: web3 and chain.rpc_url: http://127.0.0.1:8545
python experiments/run_simulation.py --config configs/experiments/baseline.yaml
```

Report per-method gas from `results/<run>/gas.json` (or `Web3Chain.gas_summary()`
if using the real chain): `registerParticipant`, `submitUpdate`,
`finalizeRound`, `distributeRewards`, and total gas per round at
`num_clients` in `{10, 50, 100}`. Multiply by a representative gas price and
an L2 vs. L1 assumption to get a $/round figure — but report **gas units as
the primary number**; a permissioned consortium chain has no meaningful
public gas price, so any dollar figure is illustrative only (see
`hardhat.config.js` comment).

Also report `communication_cost_mb()` from `models/backbones.py` per client
per round for each backbone — this is the argument for on-chain digests +
off-chain payloads, and belongs in the same table or the one right after it.

---

## Table 7 — Audit integrity checks

**Question:** does the independent auditor actually catch tampering?

This is not a numeric table — it's a checklist backed by
`tests/test_chain_and_audit.py`, which deliberately corrupts a run five
different ways (tampered artifact, forged parent hash, non-submitter
acceptance, broken weight conservation, reputation replay mismatch) and
asserts `ChainAuditor` catches each one. Report as: "5/5 injected integrity
violations detected; see supplementary test suite." Point reviewers at the
test file rather than re-deriving the checks in prose.

---

## Ablations to run if time allows (roughly priority order)

1. `alpha_cos` sweep (direction vs. magnitude weighting) at fixed attack —
   the geometric-mean-vs-additive fix documented in `reputation.py` is worth
   its own ablation row, not just a footnote.
2. GroupNorm vs BatchNorm backbone (`build_model(..., norm="group")`) under
   high non-IID skew — motivated in `models/backbones.py`'s docstring.
3. Secure aggregation ON vs OFF, showing the reputation-visibility trade-off
   from `secure_agg.py`'s docstring made concrete: detection F1 necessarily
   craters when updates are masked, because the mechanism has nothing to
   score. This ablation is what justifies calling the conflict "future work"
   rather than an oversight.
4. Single dataset (ISIC) vs. two-dataset (ISIC + ChestX-ray14) generalization,
   if time and compute allow — strengthens the paper but is not required for
   the core claim.

## Future work note (methodology, not results)

`ChainAuditor`'s checks and `orchestrator.test.js`'s checks currently
duplicate each other by hand across Python and Solidity. If this project
continues past the thesis, generating both from one specification (or at
minimum a shared fixture format) would remove the single largest
"the two didn't match" risk in the whole codebase.
