# Architecture

## Layer map

```
experiments/           you run these
  run_simulation.py       single (dataset, partition, attack, aggregator, DP) cell
  run_attack_sweep.py     Tables 1-3: aggregator x attack x malicious-fraction grid
  run_dp_sweep.py         Table 4: the privacy/detection headline trade-off

src/medchain/
  fl/                   core protocol -- pure NumPy, zero deep-learning deps
    serialization.py       canonical hashing, flatten/unflatten, delta arithmetic
    aggregation.py         FedAvg + robust baselines + OUR reputation-weighted rule
    reputation.py          cross-round trust: the paper's central mechanism
    local_trainer.py       NumPy MLP, for dependency-free CI and debugging
    client.py              NumpyClient (dev) / TorchClient (real runs) -- same interface
    server.py              FederatedServer: the round loop wiring everything together

  attacks/attacks.py     update-space, data-space, colluding, defence-aware
  data/
    partition.py            non-IID client partitioning (Dirichlet, pathological, ...)
    datasets.py              synthetic / MedMNIST / ISIC 2019 loaders
  privacy/
    dp.py                    client-level DP: clip+noise, RDP accountant
    secure_agg.py            pairwise-masking secure aggregation (simplified)
  chain/
    mock_chain.py            in-memory ledger, same API as the Solidity contracts
    client.py                web3 client for the REAL deployed contracts
  storage/
    local_cas.py             content-addressed local store (IPFS stand-in)
    ipfs.py                  real IPFS client, falls back to local_cas if no daemon
  audit/verifier.py       independent replay-and-check auditor (5 integrity checks)
  models/backbones.py     ResNet/DenseNet/EfficientNet/ViT + StateDict bridge

contracts/              Solidity, mirrors mock_chain.py method-for-method
  ParticipantRegistry.sol   membership, stake, reputation storage
  FLOrchestrator.sol        round lifecycle, hash-linked provenance
  IncentiveToken.sol        reward minting

tests/                  pytest, no GPU/node required
test/                   Hardhat/JS, requires `npm install`
```

## Why the split is where it is

**FL core is NumPy-only.** `fl/`, `attacks/`, `data/partition.py`,
`privacy/`, `chain/mock_chain.py`, `storage/`, and `audit/` import nothing
but numpy. This is a design constraint, not an accident: it means the entire
protocol — including the part carrying your novel contribution — is
unit-testable in milliseconds without a GPU, and a reviewer or replicator can
run `pip install numpy pyyaml && make demo` and watch the whole system work.
Only `models/backbones.py` and `fl/client.py`'s `TorchClient` import torch,
and they import it lazily inside `__init__` so the package still imports
cleanly on a torch-less machine.

**Simulation chain mirrors the real chain method-for-method.**
`chain/mock_chain.py` and `contracts/*.sol` implement the same operations
with the same names and argument order on purpose. Run a 200-round sweep
against `MockChain` in seconds during development; switch to
`chain/client.py` (`Web3Chain`) against a real Hardhat node for the
deployment-realism numbers (gas costs, latency) without touching
`fl/server.py` at all — it depends only on the shared interface
(`start_round`, `submit_update`, `finalize_round`, `set_reputation`, `slash`,
`mint_rewards`, `gas_summary`).

**The auditor is deliberately narrow.** `ChainAuditor` takes only a chain
object and a content store — never the orchestrator's live Python state —
because that is the only way "an untrusting third party can verify the
run" is a real claim rather than a demo convenience. See
`docs/THREAT_MODEL.md` for what each of its five checks catches.

## Data flow for one round

```
FederatedServer.run_round()
  1. sample cohort from active (non-slashed) participants
  2. each client.train(global_state) -> local StateDict
  3. compute_delta(local, global) -> flatten -> per-client update vector
  4. _apply_attack()            [malicious clients' vectors mutated]
  5. privatize_update()         [DP clip + noise, if enabled]
  6. chain.submit_update()      [hash + CID committed on-chain, payload to CAS]
  7. ledger.score_and_update()  [reputation scored from COHORT geometry, then EMA'd]
  8. chain.set_reputation()     [mirrored on-chain]
  9. get_aggregator(name)(...)  [combine accepted updates into a global delta]
 10. apply_delta() -> new global state; hash it; store it; chain.finalize_round()
 11. slash any participant whose reputation is now below the trust threshold
 12. mint_rewards() from reward_shares() [reputation x data volume]
```

Step ordering is meaningful and is documented inline in `server.py` — in
particular, DP noise (step 5) happens *before* reputation scoring (step 7),
because the aggregator in a real deployment only ever observes privatized
updates. Do not reorder this to improve detection numbers; it would produce
a claim the deployed system cannot actually make.

## Swapping in the real model

`NumpyClient` and `TorchClient` (`fl/client.py`) satisfy the same three-method
protocol (`client_id`, `num_samples`, `train(global_state) -> StateDict`), so
`FederatedServer` never needs to know which one it's holding. To move from
the dev-tier MLP to ResNet-50 on ISIC:

```python
from medchain.models.backbones import build_model, get_state
from medchain.fl.client import TorchClient
from medchain.data.datasets import isic2019_torch_datasets

train_ds, val_ds, train_labels, num_classes = isic2019_torch_datasets()
# partition train_ds indices with data/partition.py as usual, then:
clients = [TorchClient(cid, Subset(train_ds, idx), cfg, ...) for cid, idx in parts.items()]
initial_state = get_state(build_model("resnet50", num_classes))
```

Everything downstream — aggregation, reputation, chain, audit — is unchanged.

## Solidity <-> Python parity

`tests/test_chain_and_audit.py` tests semantics against `MockChain`;
`test/orchestrator.test.js` tests the same integrity properties against the
real contracts. They are not (yet) generated from one spec — if you change a
rule in one (e.g., the weight-conservation tolerance), change it in the
other and re-run both suites. This is the single highest-value thing to
automate if you extend the project past the thesis deadline (see
`docs/EXPERIMENT_MATRIX.md` "Future work" note).
