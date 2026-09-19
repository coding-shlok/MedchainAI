const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat,
} = require("docx");
const fs = require("fs");

const ASSETS = "/Users/pratikshah/Downloads/medchain_paper_assets";
const F = "Calibri";

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 100 },
    children: [new TextRun({ text, bold: true, size: 26, font: F, color: "1F3864" })],
  });
}
function h2(text) {
  return new Paragraph({
    spacing: { before: 180, after: 60 },
    children: [new TextRun({ text, bold: true, size: 22, font: F, color: "2E5496" })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 21, font: F, ...opts })],
  });
}
function b(text) {
  return new Paragraph({
    numbering: { reference: "bl", level: 0 }, spacing: { after: 70 },
    children: parseBold(text),
  });
}
function n(text) {
  return new Paragraph({
    numbering: { reference: "nl", level: 0 }, spacing: { after: 70 },
    children: parseBold(text),
  });
}
// split on **bold** markers
function parseBold(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map(s => s.startsWith("**")
    ? new TextRun({ text: s.slice(2, -2), bold: true, size: 21, font: F })
    : new TextRun({ text: s, size: 21, font: F }));
}

const body = [];

body.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "MedChain AI — Project Explainer", bold: true, size: 34, font: F, color: "1F3864" })],
}));
body.push(new Paragraph({
  spacing: { after: 200 },
  children: [new TextRun({ text: "A one-page brief for explaining the project in an interview", italics: true, size: 20, font: F, color: "666666" })],
}));

body.push(h1("One-sentence version"));
body.push(p("MedChain AI is a federated learning system that lets multiple hospitals jointly train a skin-cancer image classifier without sharing patient data, while defending against hospitals that submit bad model updates and keeping a blockchain-style audit trail of every training round."));

body.push(h1("The problem"));
body.push(b("Deep learning models for skin-lesion diagnosis need a lot of labelled images, but **no single hospital has enough**."));
body.push(b("Hospitals **cannot pool raw patient images** — privacy law (HIPAA, GDPR) and data-governance policy prevent it."));
body.push(b("Federated learning (FL) is the usual answer: each hospital trains locally and shares only **model weight updates**, not images. But that creates two new problems:"));
body.push(b("   1. A hospital could send a **corrupted or poisoned update** (by accident or on purpose) and degrade the shared model for everyone."));
body.push(b("   2. Participants have to **blindly trust** that the central server aggregated their contributions honestly."));

body.push(h1("What the system does — one training round"));
body.push(n("**Partition** — the dataset is split across simulated hospitals with realistic, uneven class distributions (non-IID)."));
body.push(n("**Local training** — each hospital fine-tunes the current global model on its own images for one epoch."));
body.push(n("**Attack hook** — for evaluation, selected 'malicious' clients corrupt their update (sign-flip, label-flip, scaling, etc.)."));
body.push(n("**Differential privacy** — every update is clipped and has calibrated noise added, bounding what the server can infer about any one hospital."));
body.push(n("**Commit to ledger** — each update's hash is recorded on a blockchain-style ledger (parent-linked hash chain)."));
body.push(n("**Reputation scoring** — the server scores each update's trustworthiness based on how it compares to the rest of the cohort, integrated over all previous rounds."));
body.push(n("**Aggregation** — updates are combined into a new global model, weighted by reputation; low-reputation updates are rejected."));
body.push(n("**Finalize + reward** — the new model hash and the accepted/rejected client list are written to the ledger; hospitals earn rewards proportional to reputation × data volume."));

body.push(h1("The key idea: reputation vs. single-round filtering"));
body.push(p("Standard Byzantine-robust rules (Krum, trimmed mean, median) judge an update only by its geometry within one round. They can't tell an attacker apart from an honest hospital that simply has a rare case mix — and they give a client that behaves for 20 rounds then defects only one round of suspicion."));
body.push(p("MedChain AI instead tracks a reputation score per client across rounds. Each round the score blends (a) cosine agreement with the robust median of all updates, and (b) whether the update's magnitude is plausible. A rare-but-honest case mix produces a stable, consistent offset; an attack produces an inconsistent or adversarial direction. Averaging over time separates the two."));

body.push(h1("Tech stack / specifics"));
body.push(b("**Task & data:** ISIC 2019 dermoscopy dataset — 25,331 images, 8 diagnostic classes. Also a binary malignant-vs-benign framing (malignant = melanoma, BCC, SCC)."));
body.push(b("**Backbones:** ResNet-18/50, DenseNet-121, EfficientNet-B0, ViT-B/16 — all ImageNet-pretrained transfer learning."));
body.push(b("**FL core:** custom federated server; NumPy-only so the protocol runs without a deep-learning framework; PyTorch only for the actual models."));
body.push(b("**Attacks implemented:** sign-flip, Gaussian noise, scaling/boosting, label-flip, and an adaptive attack that knows the defence exists."));
body.push(b("**Privacy:** client-level differential privacy (clip-then-noise) with an RDP privacy accountant."));
body.push(b("**Blockchain:** a mock chain reproducing a Solidity orchestrator contract — round hashing, lineage, staking/slashing, reward pool. Weights stored off-chain, referenced by content hash (IPFS-style)."));

body.push(h1("Results"));
body.push(b("**Binary malignant/benign:** EfficientNet-B0 — 80.6% accuracy, 0.778 macro-F1 (5 hospitals, 20 rounds)."));
body.push(b("**8-class diagnosis:** ResNet-18 — 71.5% accuracy (harder, severely imbalanced task)."));
body.push(b("**Poisoning detection:** on a controlled benchmark with 30% malicious clients, all attackers were flagged in their first round; 100% cumulative detection recall."));

body.push(h1("Why it matters (the motivating story)"));
body.push(p("ISIC has real class imbalance — melanoma is a small minority. A poisoning attack that specifically suppresses the melanoma class causes a clinically meaningful harm (missed cancers), not just an abstract accuracy drop. That makes the defence's value concrete."));

body.push(h1("What's novel here"));
body.push(p("Most papers study one piece in isolation. MedChain AI combines three in a single evaluated pipeline: non-IID federated learning on real medical images, cross-round reputation as the Byzantine defence, and blockchain-anchored round provenance."));

body.push(h1("Honest limitations (be ready for these)"));
body.push(b("The poisoning defence is validated on a synthetic benchmark (where the attacker is known); it hasn't yet been run end-to-end on the real image pipeline."));
body.push(b("Differential privacy and the attack defence haven't been evaluated jointly on real images — DP noise is known to erode the signal the defence relies on."));
body.push(b("ISIC 2019 has no patient IDs, so the train/test split is patient-agnostic and can slightly inflate accuracy."));
body.push(b("The blockchain is a local mock, not a deployed testnet."));

body.push(h1("Likely interview questions"));
body.push(b("**Why federated learning?** Data can't move — privacy law. FL moves the model to the data instead."));
body.push(b("**Why blockchain?** Auditability without a trusted central party — any participant can verify what was aggregated."));
body.push(b("**How do you catch a malicious client without punishing an unusual honest one?** Reputation integrated over time — honest = stable offset, attack = inconsistent direction."));
body.push(b("**What breaks the defence?** Differential privacy noise, and attacks that adapt to the defence — both are in the evaluation."));

const doc = new Document({
  numbering: { config: [
    { reference: "bl", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
    { reference: "nl", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
  ]},
  sections: [{
    properties: { page: { margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(`${ASSETS}/MedChainAI_Project_Explainer.docx`, buf);
  console.log("written");
});
