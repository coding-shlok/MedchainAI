# Build Plan (for Claude Code)

This repo already contains a working, tested skeleton: the full FL protocol,
reputation mechanism, mock chain, Solidity contracts, auditor, and experiment
harness run end-to-end on synthetic data (`make demo`, `make test`). What's
below is what's left to reach paper-ready results, broken into phases sized
for one Claude Code session each. Each phase has a done-condition — don't
move to the next until it's met.

Feed this file to Claude Code directly (`claude "read docs/BUILD_PLAN.md and
start Phase 1"`) rather than re-explaining the project from scratch.

---

## Phase 0 — Environment check (30 min)

- `pip install -r requirements.txt`
- `make test` — all 34 existing tests must pass with nothing but numpy/pyyaml/pytest.
- `make demo` — confirms the synthetic end-to-end pipeline still runs.
- `npm install && npm test` — confirms the Solidity contracts compile and their
  6 tests pass.

**Done when:** both test suites are green and `make demo` prints a reasonable
final accuracy (>0.85 on the synthetic config).

---

## Phase 1 — MedMNIST dev-tier validation (half day)

Purpose: shake out every experiment on cheap, fast, real (if low-res) medical
images before touching ISIC. Bugs found here cost minutes; the same bugs
found on ISIC cost GPU-hours.

- `pip install medmnist`
- Add a `configs/experiments/medmnist_dev.yaml` pointing `data.name` to
  `medmnist:pathmnist` (or `dermamnist` — closer to the ISIC skin-lesion
  task).
- Run `run_simulation.py` against it with each aggregator; confirm accuracy
  is well above chance and attacks measurably hurt the undefended baselines.
- Run a small `run_attack_sweep.py` pass (reduce `AGGREGATORS`/`ATTACKS`
  lists at the top of the script for speed) and sanity-check the CSV shape
  matches what Table 2 needs.

**Done when:** you have one real (non-synthetic) results CSV and it looks
qualitatively like the synthetic one (reputation competitive with or beating
FedAvg baseline under attack, Krum vulnerable to `colluding`).

---

## Phase 2 — ISIC 2019 data pipeline (half day, no training yet)

- Download ISIC 2019 (https://challenge.isic-archive.com/data/#2019) into
  `data/isic2019/` per the layout documented in `data/datasets.py`'s
  `isic2019_torch_datasets` docstring.
- Get `isic2019_torch_datasets()` returning real `Dataset` objects; print
  class counts and confirm the imbalance ratio matches published ISIC 2019
  statistics (melanoma is a minority class).
- Wire `TorchClient` (`fl/client.py`) to actually train one local epoch on
  one partition shard on CPU, just to confirm the shapes and loss go the
  right direction. Do not run a full federation yet.

**Done when:** `TorchClient.train()` returns a StateDict whose loss
decreased after one local epoch on a real ISIC shard, and
`communication_cost_mb()` prints a sane number for ResNet-50 (~98 MB).

---

## Phase 3 — GPU federated training loop, ISIC, no attack (1-2 days,
GPU time)

- Partition ISIC labels with `data/partition.py` (Dirichlet, alpha in
  {1.0, 0.5, 0.1}), 10-20 simulated hospitals.
- Run the full `FederatedServer` loop with `TorchClient` + ResNet-50,
  `aggregator: fedavg`, no attack, for enough rounds to see validation
  accuracy plateau (watch for overfitting given ISIC's size — consider
  early stopping on the held-out split).
- This produces the no-attack rows of **Table 1**.
- Mind the rate-limit note in your Claude usage pattern: this phase is where
  GPU/wall-clock time dominates, not model iteration — get the training loop
  right on a short run (5-10 rounds) before committing to a long one.

**Done when:** you have a real ISIC accuracy/macro-F1 number for FedAvg with
no attack, and it's stable across at least 2 seeds.

---

## Phase 4 — Attacks + defences on ISIC (2-3 days, GPU time)

- Re-run Phase 3 with each aggregator (`median`, `trimmed_mean`, `krum`,
  `multikrum`, `norm_clip`, `reputation`) under each attack
  (`sign_flip`, `scaling`, `label_flip`, `backdoor`, `colluding`,
  `reputation_aware`) at `malicious_fraction` in {0.1, 0.2, 0.3}.
- This is the GPU-expensive core of the paper (**Table 2**). Prioritize:
  `reputation` vs. every attack first (your method must be characterized
  everywhere), then fill in baseline comparisons.
- For `backdoor`: also compute attack success rate via
  `local_trainer.backdoor_success_rate` (or the torch equivalent) on a
  held-out triggered set. Backdoor ASR with *unchanged* main-task accuracy
  is the strongest evidence for why the audit trail (not just the defence)
  matters — that pairing is worth a callout figure.

**Done when:** `results/sweep_attacks/results.csv` (or its ISIC equivalent)
is populated for every cell you plan to report, with at least 2 seeds per
cell for the numbers going in the final table.

---

## Phase 5 — DP sweep + Table 3/5 ablations (1-2 days)

- Run `run_dp_sweep.py` (or its ISIC-adapted version) for **Table 4** — the
  headline privacy/detection trade-off. This is the single most important
  figure in the paper; budget real time for it.
- Run the delayed-defection sweep for **Table 3** (`attack.start_round` in
  {0, 10, 20, 30}).
- Run the relative-vs-absolute threshold ablation for **Table 5**.

**Done when:** you can state, in one sentence, the epsilon at which
detection collapses and how much `beta` buys back — that sentence is your
abstract's empirical claim.

---

## Phase 6 — Chain deployment + gas costs (half day)

- `make chain-up` (Hardhat node + deploy).
- Point one full experiment run at `chain.backend: web3` instead of
  `MockChain` (either extend `run_simulation.py` with a `--chain web3` flag,
  or swap the import in `fl/server.py` behind a config switch — keep the
  `MockChain`-shaped interface).
- Collect real gas numbers via `Web3Chain.gas_summary()` for **Table 6**.
- Run `npm test` again to reconfirm the 6 Solidity integrity tests still
  pass against whatever contract tweaks you made along the way.

**Done when:** you have real (not estimated) gas figures for
`registerParticipant`, `submitUpdate`, `finalizeRound`, `distributeRewards`
at realistic participant counts.

---

## Phase 7 — Audit demonstration + writeup assembly (1 day)

- Run `tests/test_chain_and_audit.py` and `test/orchestrator.test.js` one
  final time against the final contract/Python versions used for the paper's
  numbers — this is **Table 7**'s "5/5 detected" claim; it must be re-verified
  against the actual final code, not an earlier draft.
- Populate `paper/sections/*.md` (skeleton already in this repo) with real
  numbers, dropping in the CSVs from `results/` as the source of truth for
  every table.
- Generate figures (matplotlib) from the CSVs directly — do not hand-copy
  numbers into a plotting tool; script the figure generation into
  `paper/tables/` so a reviewer's "can you re-run this cell" request is a
  one-command answer.

**Done when:** every number in the paper traces to a specific results file
under `results/`, and both test suites are green against the exact code
version cited in the paper.

---

## Notes on pacing given Claude Pro + Sonnet rate limits

- Phases 0-2 need no GPU and no large rate-limit budget — good for solo
  short sessions.
- Phases 3-5 are GPU-bound; the Claude Code role there is mostly writing and
  debugging the run scripts, then waiting on training — batch your prompts
  accordingly rather than polling.
- Keep `results/` and `configs/experiments/*.yaml` under version control as
  you go; a lost sweep is the single most expensive thing to redo in this
  plan.
