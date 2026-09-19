# MedChain AI – Blockchain-Anchored Federated Learning for Skin-Lesion Classification

A production-ready framework combining federated learning, Byzantine-robust reputation mechanisms, and smart-contract-based audit trails for collaborative medical image classification across privacy-preserving hospital networks.

## Project Overview

**MedChain AI** addresses the core challenge of training deep learning models on sensitive patient data distributed across multiple institutions. By combining three evaluated components — non-IID federated learning on real dermoscopic images, cross-round reputation-based Byzantine defence, and gas-measured, independently auditable blockchain-anchored round provenance — the system demonstrates that collaborative, privacy-preserving, auditable medical AI is deployable rather than only theoretically sound.

### Key Results

- **Binary Malignancy Classification:** 80.6% accuracy, 0.778 macro-F1 (EfficientNet-B0, 20 federated rounds)
- **Fine-grained Diagnosis:** 71.5% accuracy on 8-class task (ResNet-18)
- **Poisoning Detection:** 100% cumulative detection recall on synthetic benchmark; all malicious clients flagged in first participation round
- **Gas Cost Validation:** Measured per-transaction costs on Hardhat EVM testbed confirm on-chain provenance is deployable at typical block gas limits

### Core Innovation

Three specific gaps in prior work are closed:
1. **Byzantine Defence on Real Data:** Reputation mechanism validated against class-imbalanced clinical dataset, not synthetic FL benchmarks
2. **Blockchain Auditability as First-Class Requirement:** Five explicit, independently checkable integrity checks (C1–C5) replace assumed properties
3. **Deployability Measurement:** Real Solidity contracts compiled and gas-measured, not left as architecture sketches

---

## Architecture

### Federated Learning Protocol (8-step round)

1. **Partition** — Data split across simulated hospitals via Dirichlet(α=0.5) non-IID partitioning
2. **Local Training** — Each hospital fine-tunes global model on local images
3. **Attack Hook** — Optional: inject sign-flip, label-flip, or scaling attacks for evaluation
4. **Differential Privacy** — Client-level clip-then-noise; RDP privacy accountant
5. **On-chain Commit** — Update hash recorded to blockchain ledger
6. **Reputation Scoring** — Cross-round trust score blending cosine agreement and magnitude plausibility
7. **Robust Aggregation** — Updates combined with reputation weighting; low-score updates rejected
8. **Finalize + Reward** — Model hash and accept/reject client list finalized; hospitals paid per reputation × data volume

### Smart Contracts (Solidity 0.8.24)

- **ParticipantRegistry:** Staking, reputation tracking, slashing mechanism
- **FLOrchestrator:** Round lifecycle management with hash-linked lineage
- **IncentiveToken:** Reward distribution driven by orchestrator's accepted-weight record

### Independent Audit Protocol

Five integrity checks (C1–C5) run offline on on-chain event log + off-chain content store:
- **C1 (Hash Integrity):** Re-hashing catches silent model substitution
- **C2 (Lineage Continuity):** Parent-hash chain validation catches history rewriting
- **C3 (Submission Validity):** Confirms aggregated updates were submitted before round finalization
- **C4 (Weight Conservation):** Accepted weights sum to one; rejected clients carry zero weight
- **C5 (Reputation Replay):** Recomputes EMA from logs; catches favouritism in scoring

---

## Dataset & Experimental Setup

**ISIC 2019:** 25,331 dermoscopic images, 8 diagnostic classes
- **Binary framing:** Malignant (MEL, BCC, SCC) vs. Benign (AK, NV, BKL, DF, VASC) — 20,265 training / 5,066 held-out
- **Partitioning:** Dirichlet(α=0.5) across 5 simulated hospitals (per-hospital sample: 166–9,073; class balance: 57.9%–98.9%)

**Backbones:** ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B/16 (ImageNet-pretrained, end-to-end fine-tuning)

---

## Repository Structure

```
medchain-ai/
├── README.md                                 # Project overview
├── .gitignore                               # Python/Node git ignores
├── research_paper/
│   ├── MedChainAI_ResearchPaper.docx        # IEEE-format paper (APA references)
│   ├── MedChainAI_Project_Explainer.docx    # One-page interview brief
│   └── RESEARCH_PAPER_NOTES.md              # Paper generation notes
├── configs/
│   └── experiments/
│       ├── isic2019_binary_vit_b16.yaml
│       └── (other configs)
├── src/medchain/
│   ├── data/
│   │   ├── datasets.py                      # ISIC 2019 loader, partitioning
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── backbones.py
│   │   └── heads.py
│   ├── fl/
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── reputation.py
│   │   ├── dp.py
│   │   └── attacks.py
│   ├── blockchain/
│   │   ├── mock_chain.py
│   │   ├── auditor.py
│   │   └── gas_measurement.py
│   └── utils/
│       └── metrics.py
├── experiments/
│   ├── run_simulation_torch.py
│   ├── run_audit.py
│   └── run_poisoning_benchmark.py
├── results/
│   ├── isic2019_binary_efficientnet_b0/
│   │   ├── final_model.pth
│   │   ├── metrics.json
│   │   └── partition_report.json
│   ├── attack_sign_flip/
│   │   └── metrics.json
│   └── gas_costs/
│       └── hardhat_measurements.json
└── paper_assets/
    ├── gen_docx.js
    ├── gen_figures.py
    ├── gen_explainer.js
    ├── fig1_pipeline.png
    ├── fig2_partition.png
    ├── fig3_reputation.png
    └── fig4_chain_architecture.png
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR-USERNAME/medchain-ai.git
cd medchain-ai

python3 -m venv venv
source venv/bin/activate
pip install torch torchvision PyYAML numpy pandas scikit-learn matplotlib
npm install --save-dev hardhat docx
```

### 2. Run Classification

```bash
# EfficientNet-B0, 20 rounds
python experiments/run_simulation_torch.py \
  --config configs/experiments/isic2019_binary_efficientnet_b0.yaml

# ViT-B/16, 50 rounds
python experiments/run_simulation_torch.py \
  --config configs/experiments/isic2019_binary_vit_b16.yaml
```

### 3. Generate Paper

```bash
cd paper_assets
python gen_figures.py
node gen_docx.js
node gen_explainer.js
```

### 4. Run Audit & Poisoning Tests

```bash
python experiments/run_audit.py --ledger results/isic2019_binary_efficientnet_b0/ledger.json
python experiments/run_poisoning_benchmark.py --num-rounds 30 --attack sign-flip --malicious-fraction 0.3
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Reputation decay (β) | 0.8 | Cross-round integration; higher = longer memory |
| Trust threshold | ~15% | Below this, clients rejected |
| DP noise multiplier | 0.1–1.0 | Privacy/accuracy trade-off sweep |
| Learning rate | 1e-4 (1e-3 for ViT) | Adam optimizer |
| Batch size | 32 (16 for ViT) | Per-hospital local training |
| Local epochs | 1 (3 for ViT) | Rounds before aggregation |
| Gradient clipping | 5.0 | L2 norm |

---

## Blockchain Deployment

### Local Testing (Hardhat)

```bash
cd blockchain/
hardhat compile
hardhat test
hardhat run scripts/measure-gas.js --network hardhat
```

**Measured Gas Costs:**
- ParticipantRegistry.register: ~169k
- FLOrchestrator.submitUpdate: ~188k (most frequent per-hospital per-round call)
- FLOrchestrator.finalizeRound: ~113k
- IncentiveToken.distributeRewards: ~122k

---

## Interview Preparation

**Your Story:**
1. **Problem:** Hospitals need collaborative AI but can't share patient data
2. **Solution:** Federated learning + reputation-based defence + blockchain audit trail
3. **Evidence:** Real results on ISIC 2019 (80.6% accuracy, 100% attack detection, measured gas costs)
4. **Honesty:** What still needs work (DP/attack joint evaluation, synthetic benchmark for now)

**Key Differentiators:**
- Reputation mechanism integrates history; single-round rules can't separate malicious from rare-but-honest
- Blockchain is not an afterthought; five explicit, testable integrity checks, real Solidity contracts, measured gas costs
- All numbers are from actual runs on real data or controlled synthetic benchmarks with ground truth

---

## Citation

```bibtex
@article{shah2024medchainai,
  title={MedChain AI: Blockchain-Anchored Federated Learning for Robust Skin-Lesion Malignancy Classification},
  author={Shah, Shlok and Mahidhar, Rajvi and Daga, Chahak and Shaikh, Shakila and Reddy, Deepti},
  year={2024},
  institution={Mukesh Patel School of Technology Management \& Engineering, SVKM's NMIMS}
}
```

---

## License

MIT — see LICENSE file.

---

**Last Updated:** September 2024  
**Status:** Published research artifact, ready for academic review and reproducibility
