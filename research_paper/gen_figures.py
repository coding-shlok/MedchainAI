import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/Users/pratikshah/Downloads/medchain_paper_assets"

# ---------------------------------------------------------------
# Fig 1: End-to-end federated round pipeline
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.2)
ax.axis("off")

stages = ["Partition\n(Dirichlet)", "Local\nTrain", "Attack\nHook", "DP\nClip+Noise",
          "On-chain\nCommit", "Reputation\nScore", "Robust\nAggregate", "Finalize +\nReward"]
colors = ["#2f6fed", "#2f6fed", "#e0453c", "#e0a13c", "#5b3fc4",
          "#1f9d55", "#1f9d55", "#2f6fed"]

n = len(stages)
box_w, box_h = 1.02, 1.15
xs = np.linspace(0.6, 9.4, n)
y = 1.1

for i, (x, label, c) in enumerate(zip(xs, stages, colors)):
    box = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                          boxstyle="round,pad=0.06,rounding_size=0.08",
                          linewidth=1.1, edgecolor=c, facecolor=c, alpha=0.15)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.3, color="#111",
             fontweight="bold")
    if i < n - 1:
        arrow = FancyArrowPatch((x + box_w/2, y), (xs[i+1] - box_w/2, y),
                                 arrowstyle="-|>", mutation_scale=10, color="#555", lw=1.1)
        ax.add_patch(arrow)

ax.text(5.0, 2.05, "One federated round", ha="center", va="center",
         fontsize=9, color="#333", style="italic")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_pipeline.png", bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------------------
# Fig 2: Per-hospital data partition (real partition_report.json values)
# ---------------------------------------------------------------
hospitals = ["hospital_00", "hospital_01", "hospital_02", "hospital_03", "hospital_04"]
benign = [3246, 143, 3818, 1123, 5152]
malignant = [553, 23, 5255, 13, 939]

fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
x = np.arange(len(hospitals))
w = 0.55
ax.bar(x, benign, w, label="Benign", color="#2f6fed")
ax.bar(x, malignant, w, bottom=benign, label="Malignant", color="#e0453c")
ax.set_xticks(x)
ax.set_xticklabels(hospitals, fontsize=8)
ax.set_ylabel("Images", fontsize=9)
ax.set_title("Non-IID partition across 5 simulated hospitals\n(Dirichlet, α = 0.5)", fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_partition.png", bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------------------
# Fig 3: Reputation trajectory, honest vs malicious (synthetic sign-flip benchmark)
# ---------------------------------------------------------------
import json
with open("/Users/pratikshah/Downloads/medchain-ai/results/attack_sign_flip/metrics.json") as f:
    m = json.load(f)
rounds = [r["round_index"] for r in m]
honest = [r["mean_reputation_honest"] for r in m]
mal = [r["mean_reputation_malicious"] for r in m]

fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=220)
ax.plot(rounds, honest, color="#1f9d55", lw=1.8, label="Honest clients (mean)")
ax.plot(rounds, mal, color="#e0453c", lw=1.8, label="Malicious clients (mean)")
ax.axhline(0.15, color="#888", lw=1.0, ls="--", label="Trust threshold region")
ax.set_xlabel("Federated round", fontsize=9)
ax.set_ylabel("Reputation score", fontsize=9)
ax.set_title("Reputation trajectory under sign-flip attack\n(synthetic benchmark, 30% malicious)", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_reputation.png", bbox_inches="tight", facecolor="white")
plt.close()

print("done")

# ---------------------------------------------------------------
# Fig 4: On-chain smart-contract architecture + independent audit
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=7.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
                        linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             color="#111", fontweight="bold")

# Hospital actor
box(0.3, 4.9, 1.7, 0.9, "Hospital\n(client)", "#555555")

# Three on-chain contracts
box(2.6, 4.9, 2.3, 0.9, "ParticipantRegistry\nstake · reputation · slash", "#2f6fed", 7.0)
box(5.4, 4.9, 2.3, 0.9, "FLOrchestrator\nstartRound · submitUpdate\nfinalizeRound · verifyLineage", "#5b3fc4", 6.6)
box(8.1, 4.9, 1.6, 0.9, "Incentive\nToken\ndistributeRewards", "#1f9d55", 6.6)

for (x1, x2) in [(2.0, 2.6), (4.9, 5.4), (7.7, 8.1)]:
    ax.add_patch(FancyArrowPatch((x1, 5.35), (x2, 5.35), arrowstyle="-|>",
                                   mutation_scale=9, color="#555", lw=1.0))

ax.text(5.0, 6.0, "On-chain contracts (Hardhat / Solidity 0.8.24)", ha="center",
         fontsize=8.3, color="#333", style="italic")

# Event log / CAS
box(1.0, 3.1, 3.6, 0.8, "Append-only event log\n(RoundStarted, UpdateSubmitted, RoundFinalized)", "#e0a13c", 6.8)
box(5.2, 3.1, 3.6, 0.8, "Off-chain content store\n(model weights, addressed by CID)", "#e0a13c", 6.8)

ax.add_patch(FancyArrowPatch((3.7, 4.9), (2.8, 3.9), arrowstyle="-|>", mutation_scale=9, color="#888", lw=1.0))
ax.add_patch(FancyArrowPatch((6.5, 4.9), (7.0, 3.9), arrowstyle="-|>", mutation_scale=9, color="#888", lw=1.0))

# Auditor
box(2.6, 1.1, 4.8, 1.1, "Independent Auditor\nC1 hash integrity · C2 lineage continuity · C3 submission validity\nC4 weight conservation · C5 reputation replay", "#e0453c", 6.6)
ax.add_patch(FancyArrowPatch((2.8, 3.1), (3.8, 2.2), arrowstyle="-|>", mutation_scale=9, color="#888", lw=1.0))
ax.add_patch(FancyArrowPatch((7.0, 3.1), (6.2, 2.2), arrowstyle="-|>", mutation_scale=9, color="#888", lw=1.0))

ax.text(5.0, 0.55, "AuditReport: PASS / FAIL, with per-round findings", ha="center",
         fontsize=8, color="#333", style="italic")
ax.add_patch(FancyArrowPatch((5.0, 1.1), (5.0, 0.78), arrowstyle="-|>", mutation_scale=9, color="#555", lw=1.0))

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_chain_architecture.png", bbox_inches="tight", facecolor="white")
plt.close()
print("fig4 done")
