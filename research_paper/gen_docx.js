const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, SectionType, Header, PageNumber, Footer, LevelFormat,
  convertInchesToTwip, TabStopType, TabStopPosition
} = require("docx");
const fs = require("fs");

const ASSETS = "/Users/pratikshah/Downloads/medchain_paper_assets";
const PAGE_W = 12240, PAGE_H = 15840; // US Letter DXA
const MARGIN = 720; // 0.5in

// ---------- small helpers ----------
function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 19, font: "Times New Roman", ...opts })],
  });
}
function heading(num, text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text: `${num}. ${text}`.toUpperCase(), bold: true, size: 20, font: "Times New Roman" })],
  });
}
function subHeading(letter, text) {
  return new Paragraph({
    spacing: { before: 140, after: 80 },
    children: [new TextRun({ text: `${letter}. ${text}`, italics: true, bold: true, size: 19, font: "Times New Roman" })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 160 },
    children: [new TextRun({ text, size: 17, font: "Times New Roman" })],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 19, font: "Times New Roman" })],
  });
}
function imageParagraph(path, width, height) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.w || 1000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.CENTER,
      children: [new TextRun({ text, bold: !!opts.header, size: 17, font: "Times New Roman" })],
    })],
  });
}
function makeTable(colWidths, rows) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, i) => new TableRow({
      children: r.map((val, j) => cell(val, { w: colWidths[j], header: i === 0 })),
    })),
  });
}
function refEntry(text) {
  return new Paragraph({
    spacing: { after: 140 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text, size: 18, font: "Times New Roman" })],
  });
}

// ---------- Title block (single column section) ----------
const titlePara = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({
    text: "MedChain AI – Blockchain-Anchored Federated Learning for Robust Skin-Lesion Malignancy Classification",
    bold: true, size: 32, font: "Times New Roman",
  })],
});

function authorCell(ord, name, email, w) {
  return new TableCell({
    borders: { top: {style: BorderStyle.NONE}, bottom: {style: BorderStyle.NONE}, left: {style: BorderStyle.NONE}, right: {style: BorderStyle.NONE} },
    width: { size: w, type: WidthType.DXA },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 20}, children: [new TextRun({text: `${ord} ${name}`, size: 20, font: "Times New Roman"})] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 0}, children: [new TextRun({text: "Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS", italics: true, size: 18, font: "Times New Roman"})] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 0}, children: [new TextRun({text: "Mumbai, India", italics: true, size: 18, font: "Times New Roman"})] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: {after: 0}, children: [new TextRun({text: email, size: 18, font: "Times New Roman"})] }),
    ],
  });
}

const AUTHOR_COL_W = 3133;
const authorTableRow1 = new Table({
  width: { size: AUTHOR_COL_W * 3, type: WidthType.DXA },
  columnWidths: [AUTHOR_COL_W, AUTHOR_COL_W, AUTHOR_COL_W],
  borders: { top: {style: BorderStyle.NONE}, bottom: {style: BorderStyle.NONE}, left: {style: BorderStyle.NONE}, right: {style: BorderStyle.NONE}, insideHorizontal: {style: BorderStyle.NONE}, insideVertical: {style: BorderStyle.NONE} },
  rows: [ new TableRow({ children: [
    authorCell("1st", "Shlok Shah", "shlokshah1310@gmail.com", AUTHOR_COL_W),
    authorCell("2nd", "Rajvi Mahidhar", "rnmahidhar@gmail.com", AUTHOR_COL_W),
    authorCell("3rd", "Chahak Daga", "chahakdaga@gmail.com", AUTHOR_COL_W),
  ]})],
});

const authorTableRow2 = new Table({
  alignment: AlignmentType.CENTER,
  width: { size: AUTHOR_COL_W * 2, type: WidthType.DXA },
  columnWidths: [AUTHOR_COL_W, AUTHOR_COL_W],
  borders: { top: {style: BorderStyle.NONE}, bottom: {style: BorderStyle.NONE}, left: {style: BorderStyle.NONE}, right: {style: BorderStyle.NONE}, insideHorizontal: {style: BorderStyle.NONE}, insideVertical: {style: BorderStyle.NONE} },
  rows: [ new TableRow({ children: [
    authorCell("4th", "Shakila Shaikh", "shakilashaikh@placeholder.com", AUTHOR_COL_W),
    authorCell("5th", "Deepti Reddy", "deeptireddy@placeholder.com", AUTHOR_COL_W),
  ]})],
});
const authorRowSpacer = new Paragraph({ spacing: { after: 120 }, children: [] });

const spacer = new Paragraph({ spacing: { after: 200 }, children: [] });

// ---------- Body (two-column section) ----------
const abstractText = "This paper presents MedChain AI, a blockchain-anchored federated learning (FL) framework for skin-lesion malignancy classification in which the blockchain layer is treated as a co-equal design objective, not an audit-log afterthought. Three Solidity smart contracts (a participant registry with staking and slashing, a round orchestrator with hash-linked lineage, and an incentive token) enforce round provenance, and an independent auditor, trusting neither the hospitals nor the aggregator, replays the on-chain event log against five explicit integrity checks. The framework also combines dermoscopic image preprocessing, transfer-learning backbones (ResNet-18, EfficientNet-B0, ViT-B/16), a cross-round reputation mechanism for Byzantine-robust aggregation, and client-level differential privacy. The trained EfficientNet-B0 binary (malignant vs. benign) classifier was evaluated on the full ISIC 2019 dataset (25,331 images, 20,265 training / 5,066 held-out, federated across 5 simulated hospitals via Dirichlet(α = 0.5) partitioning) and reached 80.6% accuracy and a macro-F1 of 0.778 after 20 rounds. A companion 8-class fine-grained diagnosis model (ResNet-18) reached 71.5% accuracy on the same cohort. The reputation-based defence was validated against a sign-flip poisoning attack on a synthetic federated benchmark, achieving 100% cumulative detection recall (60% precision, F1 = 0.75), with every malicious client flagged within its first round of participation. The smart-contract layer was compiled and executed on a Hardhat EVM testbed, and the measured per-transaction gas costs (187,778 gas on average for submitUpdate, 112,653 for finalizeRound) confirm the design is deployable rather than only theoretically auditable. Joint evaluation of the poisoning defence and differential privacy on the real dermoscopic pipeline, a fully converged Vision Transformer backbone, and a live test-net deployment are left for the next stage of this work.";

const abstractPara = new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 140 },
  children: [
    new TextRun({ text: "Abstract—", bold: true, italics: true, size: 19, font: "Times New Roman" }),
    new TextRun({ text: abstractText, italics: true, size: 19, font: "Times New Roman" }),
  ],
});

const indexTerms = new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 200 },
  children: [
    new TextRun({ text: "Index Terms—", bold: true, italics: true, size: 19, font: "Times New Roman" }),
    new TextRun({ text: "Federated learning, blockchain, smart contracts, gas cost, independent audit, dermoscopy, skin lesion classification, Byzantine-robust aggregation, reputation systems, differential privacy, EfficientNet, Vision Transformer.", italics: true, size: 19, font: "Times New Roman" }),
  ],
});

const body = [];

// I. Introduction
body.push(heading("I", "Introduction"));
body.push(bodyPara("Skin cancer screening from dermoscopic images is a task well suited to deep learning, but training one strong model usually means pooling image data from many hospitals, and patient-privacy regulation and data-governance policy routinely get in the way (Agbo et al., 2019; Price & Cohen, 2019). Federated learning (FL) offers a way around this: each hospital trains locally on its own cases and shares only model updates with a coordinating server (McMahan et al., 2017). That removes the need to centralise raw images, but it introduces two new problems. First, a participating hospital could submit a corrupted update, whether through misconfiguration or a deliberate poisoning attack (Bagdasaryan et al., 2020), and degrade the shared model for everyone else. Second, hospitals have no real way to check what the server actually did with their contribution once it has been submitted."));
body.push(bodyPara("This paper presents MedChain AI, an FL framework for skin-lesion malignancy classification built to address both problems at once. Client updates are scored every round by a cross-round reputation mechanism that tells a genuinely malicious update apart from an honest hospital with an unusual case mix, and every round is committed to a lightweight blockchain-style ledger, so the round sequence — which updates were accepted, which were rejected, and the resulting model hash — can be checked independently instead of taken on trust."));
body.push(bodyPara("The framework is evaluated on the official ISIC 2019 dermoscopic image release (Tschandl et al., 2018) — 25,331 images across 8 diagnostic classes — simulated across 5 hospitals under a non-IID Dirichlet partition. Two classification framings are trained and reported: an 8-class fine-grained diagnosis task (ResNet-18) and a coarser, clinically motivated binary malignant-versus-benign task (EfficientNet-B0), which is the easier problem and reaches a substantially higher macro-F1. A Vision Transformer (ViT-B/16) backbone is also configured for the binary task; its full-convergence result is still ongoing work and is reported separately from the headline EfficientNet-B0 number rather than mixed in with it."));
body.push(bodyPara("Separately from the image pipeline, the reputation-based poisoning defence is validated on a controlled synthetic FL benchmark, where the identity of malicious clients is known in advance. That makes it possible to report precise detection-precision and detection-recall numbers that a real, unlabelled hospital cohort simply cannot provide."));
body.push(bodyPara("This is deliberately a combined deep-learning-and-blockchain project, and the blockchain half is treated as a first-class objective rather than a logging convenience added at the end. A conventional server-side log does not really solve the problem: it still asks every hospital to trust that the coordinating server recorded and aggregated their contribution honestly, with no way to check. MedChain AI instead implements the round protocol as three Solidity smart contracts: a participant registry that holds each hospital's stake and reputation and can slash it, a round orchestrator that hash-links every round to its parent and accepts or rejects submissions on-chain, and an incentive token that mints rewards straight from the orchestrator's own accepted-weight record. Acceptance, rejection, and reward are therefore decided by contract logic, not by a party any hospital would otherwise have to trust. An independent auditor module, described in Section VI-F, then replays the resulting on-chain event log against five explicit integrity checks without touching the orchestrator's own state, and the contracts themselves are compiled and gas-measured on a real Hardhat EVM testbed (Section VIII-C) rather than left as an unevaluated design sketch."));

// II. Literature review
body.push(heading("II", "Literature Review"));
body.push(bodyPara("Federated learning for medical imaging has been explored fairly extensively as a way to train shared diagnostic models without centralising patient data, with dermoscopy, radiology, and histopathology all reported as candidate modalities (Rieke et al., 2020). The original FedAvg formulation set out the local-training-then-average protocol that most later work still builds on (McMahan et al., 2017), and subsequent surveys have catalogued the open systems and statistical challenges — communication cost, client heterogeneity, non-IID label distributions — that remain unresolved at production scale (Kairouz et al., 2021; Zhao et al., 2018). Convolutional backbones, ResNet and EfficientNet in particular, remain the dominant choice for transfer learning on dermoscopic images (He et al., 2016; Tan & Le, 2019), thanks to strong ImageNet-pretrained feature extractors and comparatively modest compute requirements next to Vision Transformers (Dosovitskiy et al., 2021). Attention-based architectures have more recently been explored specifically for dermoscopic lesion representations (Yan et al., 2019), building on the earlier demonstration that deep convolutional networks can match dermatologist-level performance on skin cancer classification (Esteva et al., 2017)."));
body.push(bodyPara("Byzantine-robust aggregation has mostly been studied as single-round geometric filtering: coordinate-wise median and trimmed-mean rules, along with Krum-style distance-based rules, identify outlying updates within one round's cohort (Yin et al., 2018; Blanchard et al., 2017). These rules are blind to history, so a client that behaves honestly for many rounds and defects once gets only momentary suspicion. Reputation-based approaches integrate a trust signal across rounds instead (Fung et al., 2018), and have been shown to tell a persistent attacker apart from a client whose data distribution is merely unusual but stable — the two cases a single-round rule cannot separate. Heterogeneity-aware optimisation methods such as FedProx address a related but different problem, stabilising convergence under non-IID client data rather than defending against an adversarial one (Li et al., 2020), and are complementary to, rather than a substitute for, an explicit Byzantine defence."));
body.push(bodyPara("Differential privacy (DP) applied at the client level — clipping and noising the aggregate update rather than individual gradients — is the standard way to bound what an aggregator can infer about any one participant's contribution (McMahan et al., 2018), building on the original formulation of differentially private stochastic training (Abadi et al., 2016). Secure aggregation protocols pursue a related but different guarantee, cryptographically hiding individual updates from the aggregator entirely instead of bounding statistical leakage (Bonawitz et al., 2017). It is well established that DP noise is isotropic and compresses the geometric gap Byzantine-robust rules rely on, so the two defences interact rather than compose independently; a poisoning attack can also be analysed and strengthened under an adversarial-lens framework that assumes the attacker already knows the deployed defence (Bhagoji et al., 2019)."));
body.push(bodyPara("Separately, blockchain-anchored FL has been proposed as a way to make the aggregation process auditable (Nguyen et al., 2021): committing round hashes and accepted/rejected client sets to a ledger lets any participant verify that the server aggregated exactly what it claims, instead of trusting the server's own logs. A recurring objection to this idea, raised whenever it is evaluated as a systems contribution rather than a protocol sketch, is deployability — storing full model weights on-chain is computationally infeasible at any realistic gas cost, so the usual fix is to store only a content digest on-chain and keep the weights in off-chain, content-addressable storage such as the InterPlanetary File System, referenced by hash (Benet, 2014). Empirical gas-cost analysis of deployed smart contracts is the standard way such designs get shown to be practical rather than only theoretically auditable (Chen et al., 2017), and independent or formal verification of the resulting ledger — checking hash integrity, lineage continuity, and conservation properties without trusting whoever produced the ledger — is increasingly treated as a requirement in its own right rather than an assumed property of \"using a blockchain\" (Bhargavan et al., 2016). Smart-contract-based incentive mechanisms have also been proposed to turn a reputation or contribution score directly into an on-chain reward, closing the loop between behaviour and payment within the same trust boundary as the audit itself (Weng et al., 2019). MedChain AI combines these threads — non-IID FL on real dermoscopic data (Goyal et al., 2020; Zhao et al., 2018), cross-round reputation as the Byzantine defence (Fung et al., 2018), and gas-measured, independently auditable ledger-anchored provenance (Nguyen et al., 2021; Benet, 2014; Chen et al., 2017; Bhargavan et al., 2016) — within one evaluated pipeline, rather than studying them separately."));

body.push(subHeading("A", "Research Gap and Novelty"));
body.push(bodyPara("Read together, the literature above leaves three fairly specific gaps rather than one general one. First, federated medical-imaging studies (Rieke et al., 2020) typically report only accuracy or macro-F1 under an honest cohort and do not evaluate a Byzantine defence at all, so their robustness to a corrupted participant is untested rather than simply unreported. Second, where a Byzantine-robust or reputation-based defence is proposed and evaluated (Yin et al., 2018; Fung et al., 2018), it is almost always benchmarked on synthetic or generic-vision FL data rather than a real, severely class-imbalanced clinical dataset, leaving open whether the honest/malicious separation survives contact with the label skew a real hospital cohort actually produces (Section V, Table I), where a rare-but-honest case mix and a genuine attack can look geometrically similar. Third, blockchain-anchored FL proposals (Nguyen et al., 2021; Benet, 2014) are consistently presented at the level of architecture — a ledger design, an incentive scheme — without the two pieces of evidence a systems reviewer would actually ask for: measured gas cost on real, compiled contracts, and an independent, claim-by-claim audit protocol rather than an assumed property of \"using a blockchain.\" Nothing surveyed here closes all three gaps in one evaluated system, and closing only one or two still leaves an adoption-blocking objection standing: a hospital cohort is not going to federate on a system whose defence is untested against real data, or whose audit trail still has to be taken on trust."));
body.push(bodyPara("This is the need this paper is addressing: a federated skin-lesion classifier is only as trustworthy as its two weakest points — the possibility of a corrupted participant, and the possibility of a dishonest coordinator — and hardening only one of the two does not actually remove the reason a hospital would decline to participate. MedChain AI's novelty is therefore the specific combination of three contributions evaluated together, each closing one of the gaps above, rather than any single piece on its own: (i) a cross-round reputation defence evaluated with detection precision and recall against a known attacker set (Section VIII-B), not just a before/after accuracy comparison; (ii) that defence and the classification backbones evaluated on a real, non-IID-partitioned clinical imaging dataset with genuine class imbalance (Sections IV–V), rather than a synthetic or generic-vision FL benchmark; and (iii) a blockchain layer whose deployability is demonstrated, not assumed — three Solidity contracts compiled and gas-measured on a real EVM testbed (Table IV) and checked against five independently verifiable integrity claims (Section VI-F), so the audit guarantee does not rely on the paper's own description of it."));

// III. Problem statement
body.push(heading("III", "Problem Statement"));
body.push(subHeading("A", "Problem Statement"));
body.push(bodyPara("Training a skin-lesion malignancy classifier across multiple hospitals without centralising patient images calls for a federated protocol that (i) tolerates the severe class imbalance and non-IID case-mix differences that come from each hospital's patient population being different, (ii) detects and excludes corrupted or malicious client updates without penalising honest hospitals whose case mix is simply rare, and (iii) leaves a verifiable record of how each round was aggregated. Existing single-model or single-round-defence approaches do not address all three together."));
body.push(bodyPara("Requirement (iii) is not a secondary concern that an application log can quietly satisfy, and this paper treats it with the same weight as the classification problem. A federated round, as normally implemented, asks every participating hospital to trust a single coordinating server — that it recorded each hospital's update, scored it fairly, aggregated exactly the accepted set it claims to have aggregated, and paid rewards accordingly — with no way for a hospital, or an outside regulator, to check any of that afterwards. The problem the blockchain component addresses is therefore a different one from the classification problem: how to make round provenance, acceptance, and reward verifiable by a party that trusts neither the hospitals nor the aggregator, at a gas cost that keeps the design deployable rather than merely theoretical. Sections VI-C, VI-F, and VIII-C develop and evaluate this requirement to the same standard of evidence as the classification results."));

// IV. Datasets
body.push(heading("IV", "Datasets"));
body.push(bodyPara("The framework is evaluated on the official ISIC 2019 Challenge training release: 25,331 dermoscopic images spanning eight diagnostic categories — melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF), vascular lesion (VASC), and squamous cell carcinoma (SCC)."));
body.push(subHeading("A", "Binary Malignancy Framing"));
body.push(bodyPara("A clinically motivated binary label is derived by grouping MEL, BCC, and SCC — the three categories that are actual cancers — as malignant, and every remaining category (including AK, a precancerous but conventionally benign-grouped condition) as benign. This is a coarser, different task from the 8-class diagnosis problem and is reported as a separate number throughout, never as an improved score on the same benchmark."));
body.push(subHeading("B", "Train / Held-out Split"));
body.push(bodyPara("A random 80/20 split yields 20,265 training images and 5,066 held-out images. The split is patient-agnostic: ISIC 2019 ships no reliable patient identifier, so images from the same patient can in principle land on both sides of the split. This is a real limitation of the dataset rather than the pipeline, and it is treated as such in the discussion below."));

// V. Pre-processing (formerly Federated Partitioning and Data Distribution)
body.push(heading("V", "Pre-processing"));
body.push(bodyPara("The 20,265 training images are partitioned across 5 simulated hospitals using a Dirichlet(α = 0.5) label-distribution split, the standard non-IID benchmark configuration. α = 0.5 produces meaningful but not pathological skew: some hospitals see close to a balanced case mix while others see a heavily skewed one, mirroring real inter-hospital referral patterns. Table I and Fig. 2 report the resulting per-hospital sample counts and class balance actually produced by this partition."));
body.push(imageParagraph(`${ASSETS}/fig2_partition.png`, 460, 216));
body.push(caption("Fig. 2. Per-hospital class distribution after Dirichlet(α = 0.5) partitioning of the 20,265 training images across 5 simulated hospitals."));
body.push(makeTable([2400,1700,1700,1700,1900],[
  ["Hospital","Samples","Benign","Malignant","Max class share"],
  ["hospital_00","3,799","3,246","553","85.4%"],
  ["hospital_01","166","143","23","86.1%"],
  ["hospital_02","9,073","3,818","5,255","57.9%"],
  ["hospital_03","1,136","1,123","13","98.9%"],
  ["hospital_04","6,091","5,152","939","84.6%"],
]));
body.push(caption("TABLE I. Per-hospital partition statistics (binary framing, Dirichlet α = 0.5)."));
body.push(bodyPara("This confirms that label skew and low sample count vary substantially across hospitals — hospital_01 sees only 166 images while hospital_02 sees 9,073 — which is the intended stress test for both the classification backbone and the reputation mechanism described in Section VI-E."));

// VI. Proposed system architecture
body.push(heading("VI", "Proposed System Architecture"));
body.push(bodyPara("MedChain AI structures every federated round as a fixed nine-step protocol, illustrated in Fig. 1: partition the cohort, train locally on the current global state, apply the (optional) attack hook to any malicious client's update for evaluation purposes, clip and noise every update under differential privacy, commit each client's update hash to the ledger, score reputation from cohort geometry, aggregate updates under a reputation-weighted rule, finalize the round with the resulting model hash and accepted/rejected sets, and mint rewards from reputation × contributed data volume."));
body.push(imageParagraph(`${ASSETS}/fig1_pipeline.png`, 460, 158));
body.push(caption("Fig. 1. End-to-end federated round protocol implemented by MedChain AI."));
body.push(subHeading("A", "Backbones"));
body.push(bodyPara("Five ImageNet-pretrained backbones are supported through a common state-dict bridge: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, and ViT-B/16. All are fine-tuned end-to-end with a replaced classification head sized to the target task (2 classes for the binary framing, 8 for fine-grained diagnosis)."));
body.push(subHeading("B", "What Gets Federated"));
body.push(bodyPara("By default, the shared state includes both trainable parameters and BatchNorm running statistics. BatchNorm statistics are known to leak information about a client's local input distribution — a documented FL privacy consideration — but excluding them measurably hurts accuracy under the non-IID partition used here. The framework supports a privacy-preferring GroupNorm configuration as an alternative; the results in this paper use the default BatchNorm configuration, and this trade-off is stated explicitly rather than left implicit."));
body.push(subHeading("C", "Ledger and Provenance"));
body.push(bodyPara("Round provenance is implemented as three Solidity smart contracts, not as an internal log format. ParticipantRegistry holds each hospital's stake (posted as payable Ether at registration), its current reputation in basis points, and an active/slashed status. FLOrchestrator owns the round lifecycle — startRound, submitUpdate, and finalizeRound — and hash-links every round to its parent, so the round sequence forms a tamper-evident chain rather than a mutable table. IncentiveToken mints rewards through distributeRewards, driven only by the orchestrator's own accepted-weight record. Each round is linked to its parent via this hash chain, and per-round content — the aggregated model hash, accepted/rejected client identifiers, and per-client reward — is written to an append-only lineage record that FLOrchestrator's own verifyLineage function can walk to confirm no round has been rewritten."));
body.push(bodyPara("A key design choice, and the usual reason blockchain-FL proposals get judged impractical, is what is kept off-chain. Storing full model weights on-chain would cost on the order of 10^9 gas per round at typical parameter counts, which is not deployable at any realistic gas price, so MedChain AI instead commits only a 32-byte digest and a short content identifier (CID) to FLOrchestrator, while the weights themselves live in off-chain, content-addressable storage referenced by that CID — an IPFS-style separation that keeps the on-chain footprint small and, as Section VIII-C shows, measurable rather than assumed."));
body.push(bodyPara("The current implementation runs this exact contract interface two ways: a local mock chain (Python, used for the 20-round classification experiments in Section VIII-A because it reproduces every method and event of the Solidity contracts with the same names and argument order, for fast iteration) and the real, compiled Solidity contracts executed on a Hardhat EVM testbed for correctness and gas measurement (Section VIII-C). Deployment against a public test-net is prepared but not yet exercised, and is stated as a limitation in Section VIII-F."));
body.push(subHeading("D", "Model Training and Classification"));
body.push(bodyPara("All backbones are fine-tuned with the Adam optimizer at a learning rate of 1e-4, weight decay 1e-4, batch size 32, one local epoch per round, gradient clipping at norm 5.0, and a cosine learning-rate schedule decayed over the run. Class-balanced cross-entropy loss — inverse-frequency weighting computed from each client's own local label counts, in the spirit of Cui et al. (2019) — is used throughout, since a rare class that is 5× under-represented at a given hospital would otherwise be 5× under-weighted in that hospital's local gradient, degrading macro-F1 even when raw accuracy looks acceptable."));
body.push(bodyPara("Two classification framings are trained and reported separately. The 8-class fine-grained diagnosis task uses a ResNet-18 backbone. The binary malignant-versus-benign task — an easier, more balanced framing (roughly 67/33 class split versus severe 9-way skew) — uses an EfficientNet-B0 backbone, chosen for its typically stronger transfer-learning performance relative to ResNet-18 at comparable parameter count. Both configurations are trained for 20 federated rounds under the reputation aggregator with no active attack, establishing the honest-cohort baseline reported in Section VIII."));
body.push(subHeading("E", "Byzantine-Robust Reputation and Poisoning Defence"));
body.push(bodyPara("A single-round robust rule judges an update only by its geometry within that round's cohort, so it treats a client that attacks once after twenty honest rounds the same as one that attacks continuously, and it penalises an honest hospital with a rare case mix exactly as harshly as an attacker, every round, forever. MedChain AI instead integrates behaviour over time: each client i maintains a reputation R_i ∈ [0,1], updated every round as R_i ← clip(β·R_i + (1−β)·s_i, 0, 1), where s_i blends (a) cosine agreement between the client's update and the coordinate-wise median of the cohort — a Byzantine-robust reference that an attacker cannot arbitrarily drag, unlike the mean — and (b) a log-space magnitude-plausibility term that penalises updates that are implausibly large or small relative to the cohort's median norm."));
body.push(bodyPara("DP noise is isotropic: it pushes every pairwise cosine toward zero and compresses the honest/malicious separation the reputation score depends on. Two mitigations are built into the scoring. One is robust standardisation using the cohort's own median and median-absolute-deviation, rather than a fixed absolute threshold, so a uniform DP-induced shift largely cancels out. The other is a higher EMA decay β, which trades detection latency for detection accuracy by averaging noise across more rounds. Consistent with an honest privacy threat model, DP noise is applied to every update before reputation scoring, not after, so the aggregator never observes a clean update, only a privatised one — the harder and more realistic ordering."));
body.push(bodyPara("Because the real ISIC 2019 cohort has no labelled ground-truth attacker, the reputation mechanism is validated on a controlled synthetic FL benchmark (10 clients, Dirichlet α = 0.5, 3 of 10 clients — 30% — mounting an update-space sign-flip attack from round 0, 30 rounds total) where detection precision and recall against the known attacker set can be computed exactly. Fig. 3 shows the resulting reputation trajectory; the corresponding detection-performance table and its interpretation are reported in Section VIII-B."));
body.push(imageParagraph(`${ASSETS}/fig3_reputation.png`, 400, 175));
body.push(caption("Fig. 3. Mean reputation of honest versus malicious clients across 30 rounds under a sign-flip attack. Malicious clients are suppressed immediately but reputation partially recovers as the attack's relative signal weakens near convergence."));

body.push(subHeading("F", "Independent On-Chain Audit Protocol"));
body.push(bodyPara("Reputation scoring (Section VI-E) still relies on the aggregator to compute and report scores honestly. The audit protocol removes that assumption: an independent auditor module takes only the on-chain event log and the off-chain content store as input, never the orchestrator's in-memory state, and is designed to keep working even if every other Python object in the system were deleted. The point is that a regulator who trusts neither the hospitals nor the aggregator can still verify what happened, using only public, append-only inputs. Fig. 4 shows the full architecture: hospitals interact with the three contracts of Section VI-C, which emit events and off-chain CIDs that the auditor independently replays."));
body.push(imageParagraph(`${ASSETS}/fig4_chain_architecture.png`, 430, 226));
body.push(caption("Fig. 4. On-chain smart-contract architecture and the independent audit path: the auditor consumes only the public event log and content store, never the orchestrator's internal state."));
body.push(bodyPara("The auditor runs five checks, each mapped to a specific integrity claim and each with a corresponding Hardhat test. C1 (hash integrity) confirms every stored artifact re-hashes to its committed digest, catching silent substitution of weights. C2 (lineage continuity) confirms each round's parent hash equals the previous round's model hash, catching history rewriting. C3 (submission validity) confirms every aggregated update was submitted before its round was finalized, catching an aggregator inserting its own fabricated update. C4 (weight conservation) confirms accepted weights sum to one and rejected clients carry zero weight, catching silent inclusion of an excluded party. C5 (reputation replay) recomputes the reputation EMA from the logged per-round scores and confirms it reproduces the logged reputations, catching favouritism in scoring that would not otherwise surface in the final numbers. Each check is tied to a specific, testable claim, so that a claim not checkable this way is simply not made in this paper."));

// VII. Experimental setup (formerly IX)
body.push(heading("VII", "Experimental Setup and Evaluation Metrics"));
body.push(bodyPara("Classification performance is reported using accuracy, macro-averaged F1-score, and cross-entropy loss on the held-out validation set after each round. Macro-F1 is reported alongside accuracy because it is unaffected by the majority-class bias that accuracy alone can hide under class imbalance — a model that always predicts the majority class scores well on accuracy but poorly on macro-F1."));
body.push(bodyPara("Poisoning-detection performance is reported using precision, recall, and F1 against the known malicious-client set, computed two ways: a final-round snapshot (which clients are currently below the acceptance threshold) and a cumulative, ever-flagged view (which clients were ever rejected across the run). These capture different operational questions, so both are reported rather than only the more favourable one."));
body.push(bodyPara("At the current stage, the poisoning-detection evaluation (Section VI-E) is performed on a synthetic, labelled FL benchmark rather than directly on the ISIC 2019 image pipeline, and differential privacy is implemented but was disabled (ε → ∞ equivalent) for the classification runs reported in Section VIII. Combining attack simulation, DP, and the real dermoscopic pipeline in one run is the next stage of development — consistent with the framework's design, but not yet exercised end-to-end."));

// VIII. Results and Analysis (formerly X. Results)
body.push(heading("VIII", "Results and Analysis"));
body.push(subHeading("A", "Classification Performance"));
body.push(makeTable([2600,2000,1900,1900,1900],[
  ["Task","Backbone","Rounds","Accuracy","Macro-F1"],
  ["8-class diagnosis","ResNet-18","20","71.5%","0.498"],
  ["Binary malignancy","EfficientNet-B0","20","80.6%","0.778"],
]));
body.push(caption("TABLE II. Final-round classification performance, honest cohort (no active attack), ISIC 2019."));
body.push(bodyPara("The binary framing substantially outperforms the 8-class task on both metrics, which fits with it being a strictly easier, better-balanced problem rather than an improved score on the same benchmark — the two numbers are not directly comparable and are reported side by side for that reason. The gap between accuracy (80.6%) and macro-F1 (0.778) on the binary task shows that the remaining ~67/33 class imbalance still measurably affects the minority (malignant) class even under class-balanced loss weighting."));
body.push(subHeading("B", "Poisoning Detection"));
body.push(bodyPara("Table III reports poisoning-detection performance for the synthetic sign-flip benchmark described in Section VI-E; the corresponding reputation trajectory is shown in Fig. 3."));
body.push(makeTable([3300,2000,2000,2000],[
  ["Metric","Final-snapshot","Cumulative (ever-flagged)",""],
  ["Precision","0.00","0.60",""],
  ["Recall","0.00","1.00",""],
  ["F1-score","0.00","0.75",""],
]));
body.push(caption("TABLE III. Poisoning-detection performance, synthetic sign-flip benchmark (3 of 10 clients malicious)."));
body.push(bodyPara("All three malicious clients were flagged within their very first round of participation (rounds_to_detect = 0 for each). The final-round snapshot shows zero detection because, by round 29, reputation for the malicious clients had drifted back above the acceptance threshold as the sign-flip attack's relative geometric signal weakened near convergence — a real, reported limitation rather than something hidden in the final number. The cumulative, ever-flagged view (recall = 1.0, precision = 0.6) is the metric that reflects what an operator monitoring the ledger's rejection history would actually see."));
body.push(subHeading("C", "Ledger Provenance"));
body.push(bodyPara("Every round of both classification runs produced a verifiable lineage record — a parent-linked round hash, the aggregated model hash, and the accepted/rejected client sets — confirming that the commit-then-finalize protocol (Section VI-C) executes correctly end-to-end on real training runs, whether or not an attack is active. Across the 20-round binary classification run, the mock chain recorded 249 on-chain transactions in total, spanning participant registration, per-round submissions, reputation updates, finalization, and reward distribution."));
body.push(bodyPara("To confirm the design is deployable rather than only theoretically auditable, the three Solidity contracts (Section VI-C) were compiled with Solidity 0.8.24 and executed on a Hardhat EVM testbed with gas reporting enabled, driven by the correctness test suite described in Section VI-F. Table IV reports the resulting measured, per-transaction gas costs — not the fixed illustrative constants the fast Python mock chain used for the 20-round training run above, but actual EVM execution costs of the real contracts."));
body.push(makeTable([2900,2400,1700,1700],[
  ["Contract · Method","Avg. gas","Calls","Min–Max"],
  ["ParticipantRegistry.register","169,130","21","163,394–180,578"],
  ["ParticipantRegistry.slash","34,663","1","—"],
  ["FLOrchestrator.startRound","105,659","7","96,617–136,813"],
  ["FLOrchestrator.submitUpdate","187,778","7","173,117–190,229"],
  ["FLOrchestrator.finalizeRound","112,653","3","112,645–112,657"],
  ["IncentiveToken.distributeRewards","121,797","1","—"],
]));
body.push(caption("TABLE IV. Measured per-transaction gas cost, real Solidity contracts on a Hardhat EVM testbed (Solc 0.8.24, optimizer enabled, 200 runs)."));
body.push(bodyPara("Contract deployment itself costs 1,116,419 gas for FLOrchestrator, 745,592 gas for ParticipantRegistry, and 489,097 gas for IncentiveToken — each under 2% of a typical 60,000,000 gas block limit, and each paid once per system lifetime rather than once per round. submitUpdate is the single most expensive recurring call, since every participating hospital calls it every round; at roughly 188,000 gas it is still a small fraction of typical block gas limits even at moderate cohort sizes, and none of it scales with model size, because the transferred weights themselves are never on-chain (Section VI-C). These are measured costs from actually compiling and running the contracts, not projected or assumed figures, and they are the number a systems reviewer tends to ask for first when judging whether a blockchain-FL design is deployable."));
body.push(subHeading("D", "Current Development Status"));
body.push(bodyPara("A ViT-B/16 configuration for the binary task is prepared and undergoing training with an extended 50-round schedule and ViT-appropriate hyperparameters (higher learning rate, smaller batch size, additional local epochs per round) meant to give the transformer backbone enough rounds to converge. Its result is not yet final and is therefore not included in Table II; an initial 20-round run underperformed EfficientNet-B0 (80.1% vs. 80.6% accuracy), which is reported here for completeness rather than left out, and motivates the extended run rather than a claim of improvement."));
body.push(subHeading("E", "Discussion"));
body.push(bodyPara("The results show that a federated pipeline combining non-IID partitioning, transfer-learning backbones, cross-round reputation, and ledger-anchored provenance is workable end-to-end on a real, imbalanced medical imaging dataset. The binary malignancy classifier's 80.6% accuracy is a meaningful signal given the federated, non-IID setting (Table I shows hospital-to-hospital class share ranging from 57.9% to 98.9%), though the macro-F1 of 0.778 shows there is room for improvement on the minority malignant class specifically."));
body.push(bodyPara("The reputation defence's behaviour is worth reporting honestly rather than as an unqualified success: it detects every attacker immediately, but a single-attack-type, static-parameter defence loses final-round visibility as the attack's relative signal fades near convergence. That fits the mechanism's own design rationale (Section VI-E) — reputation integrates history to avoid one-round blindness for honest clients, but the same smoothing lets a suppressed reputation partially recover once the attack itself weakens in effect. Reporting both the final-snapshot and cumulative metrics, instead of only the more favourable cumulative number, is a deliberate choice."));
body.push(bodyPara("That the poisoning defence is currently validated on a synthetic, labelled benchmark rather than the real ISIC 2019 pipeline is a real scope boundary of this work, not an oversight: ISIC 2019 carries no ground-truth attacker label against which detection precision and recall could be computed directly, so the synthetic benchmark is the only setting where those numbers currently mean anything. The synthetic result shows the mechanism works as designed; carrying it onto the real image pipeline, where an attacker's update is a real hospital's SGD gradient rather than a synthetic transformation, is future work, addressed in Section VIII-F."));
body.push(bodyPara("The blockchain component is evaluated to the same standard as the classification component on purpose: Table IV reports measured, not assumed, gas costs from the real compiled contracts, and Section VI-F's five-check audit protocol is testable claim by claim rather than a general assertion that \"the system uses a blockchain.\" The resulting picture is that on-chain provenance is cheap relative to typical block gas limits (submitUpdate, the most frequent call, costs under 190,000 gas even before any optimisation pass), precisely because model weights never touch the chain — the on-chain/off-chain split of Section VI-C is what makes the audit guarantee affordable and not just correct in principle. That is the direct answer to why a blockchain is used here rather than a centralised log: a log has to be trusted, while FLOrchestrator's hash-linked lineage and the independent auditor's five checks can be verified by a party that trusts neither the hospitals nor the server, at a measured and fairly modest gas cost."));
body.push(subHeading("F", "Limitations and Future Work"));
body.push(bodyPara("The current implementation is an initial stage of the proposed framework and has a few limitations worth stating plainly."));
[
  "Poisoning detection (Table III) is validated on a synthetic FL benchmark with a known attacker set; it has not yet been run against the real ISIC 2019 image pipeline, where updates come from real hospital SGD rather than a synthetic transformation.",
  "Differential privacy is implemented (client-level clip-then-noise, RDP accountant) but was disabled for the classification results in Table II; its known erosion of the reputation signal (Section VI-E) has not yet been quantified jointly with detection performance on real images.",
  "The ISIC 2019 train/held-out split is patient-agnostic because the dataset ships no reliable patient identifier, so recordings from the same patient can appear on both sides of the split — a real limitation of the dataset rather than the pipeline, and one that can inflate held-out accuracy.",
  "The ViT-B/16 backbone's binary-task result is not yet final; an initial 20-round run underperformed EfficientNet-B0, and a longer, ViT-tuned run is in progress at the time of writing.",
  "The 20-round classification runs (Table II) use the fast Python mock chain, not the compiled Solidity contracts directly, for iteration speed; the real contracts are compiled, correctness-tested, and gas-measured on a Hardhat EVM testbed (Table IV) but have not yet been deployed to a public or private test-net under real network latency and multi-node consensus.",
].forEach(t => body.push(bullet(t)));
body.push(bodyPara("Future work will focus on completing the extended ViT-B/16 run and reporting its converged result alongside EfficientNet-B0; porting the sign-flip and label-flip poisoning attacks onto the real ISIC 2019 federated pipeline so detection metrics can be computed on real hospital updates; sweeping the differential-privacy noise multiplier against reputation-detection accuracy to characterise the accuracy/privacy/robustness trade-off; and deploying the ledger component to a public test-net to validate on-chain provenance outside the local mock-chain environment."));

// IX. Conclusion (formerly XIII)
body.push(heading("IX", "Conclusion"));
body.push(bodyPara("This paper presented MedChain AI, a blockchain-anchored federated learning framework for skin-lesion malignancy classification that combines a non-IID-partitioned real dermoscopic dataset, transfer-learning backbones, cross-round reputation-based Byzantine defence, client-level differential privacy, and a smart-contract-enforced, independently auditable round-provenance layer within one evaluated pipeline. The binary malignant-versus-benign classifier reached 80.6% accuracy and a macro-F1 of 0.778 on the full ISIC 2019 dataset federated across 5 simulated hospitals, and the reputation defence achieved 100% cumulative detection recall against a labelled poisoning attack on a controlled benchmark, with every attacker flagged in its first round of participation. The blockchain layer was evaluated to the same standard: three Solidity contracts were compiled and executed on a Hardhat EVM testbed, an independent auditor validated five distinct integrity claims against the resulting on-chain event log, and the measured gas costs confirm the on-chain/off-chain design is deployable rather than only theoretically sound."));
body.push(bodyPara("The poisoning defence and privacy components are validated independently at the current stage; combining them directly on the real image pipeline, together with a fully converged Vision Transformer backbone, is the next stage of this work. The overall goal is a federated, auditable, Byzantine-robust decision-support pipeline that can assist, not replace, clinical dermatological assessment across institutions that cannot share raw patient data."));

// References (APA 7th edition, alphabetical by first author)
body.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  alignment: AlignmentType.CENTER,
  spacing: { before: 220, after: 140 },
  children: [new TextRun({ text: "References", bold: true, size: 20, font: "Times New Roman" })],
}));
const refs = [
  "Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (pp. 308–318). ACM.",
  "Agbo, C. C., Mahmoud, Q. H., & Eklund, J. M. (2019). Blockchain technology in healthcare: A systematic review. Healthcare, 7(2), 56.",
  "Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020). How to backdoor federated learning. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (pp. 2938–2948). PMLR.",
  "Benet, J. (2014). IPFS – Content addressed, versioned, P2P file system (arXiv:1407.3561). arXiv.",
  "Bhagoji, A. N., Chakraborty, S., Mittal, P., & Calo, S. (2019). Analyzing federated learning through an adversarial lens. In Proceedings of the 36th International Conference on Machine Learning (pp. 634–643). PMLR.",
  "Bhargavan, K., Delignat-Lavaud, A., Fournet, C., Gollamudi, A., Gonthier, G., Kobeissi, N., Kulatova, N., Rastogi, A., Sibut-Pinote, T., Swamy, N., & Zanella-Béguelin, S. (2016). Formal verification of smart contracts: Short paper. In Proceedings of the 2016 ACM Workshop on Programming Languages and Analysis for Security (pp. 91–96). ACM.",
  "Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems 30 (pp. 119–129). Curran Associates.",
  "Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A., & Seth, K. (2017). Practical secure aggregation for privacy-preserving machine learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (pp. 1175–1191). ACM.",
  "Chen, T., Li, X., Luo, X., & Zhang, X. (2017). Under-optimized smart contracts devour your money. In 2017 IEEE 24th International Conference on Software Analysis, Evolution and Reengineering (SANER) (pp. 442–446). IEEE.",
  "Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9268–9277). IEEE.",
  "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations.",
  "Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115–118.",
  "Fung, C., Yoon, C. J. M., & Beschastnikh, I. (2018). Mitigating sybils in federated learning poisoning (arXiv:1808.04866). arXiv.",
  "Goyal, M., Knackstedt, T., Yan, S., & Hassanpour, S. (2020). Artificial intelligence-based image classification methods for diagnosis of skin cancer: Challenges and opportunities. Computers in Biology and Medicine, 127, 104065.",
  "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778). IEEE.",
  "Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... Zhao, S. (2021). Advances and open problems in federated learning. Foundations and Trends in Machine Learning, 14(1–2), 1–210.",
  "Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems, 2, 429–450.",
  "McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (pp. 1273–1282). PMLR.",
  "McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. International Conference on Learning Representations.",
  "Nguyen, D. C., Ding, M., Pham, Q.-V., Pathirana, P. N., Le, L. B., Seneviratne, A., Li, J., Niyato, D., & Poor, H. V. (2021). Federated learning meets blockchain in edge computing: Opportunities and challenges. IEEE Internet of Things Journal, 8(16), 12806–12825.",
  "Price, W. N., & Cohen, I. G. (2019). Privacy in the age of medical big data. Nature Medicine, 25(1), 37–43.",
  "Rieke, N., Hancox, J., Li, W., Milletarì, F., Roth, H. R., Albarqouni, S., Bakas, S., Galtier, M. N., Landman, B. A., Maier-Hein, K., Ourselin, S., Sheller, M., Summers, R. M., Trask, A., Xu, D., Baust, M., & Cardoso, M. J. (2020). The future of digital health with federated learning. npj Digital Medicine, 3, Article 119.",
  "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the 36th International Conference on Machine Learning (pp. 6105–6114). PMLR.",
  "Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, Article 180161.",
  "Weng, J., Weng, J., Li, M., Zhang, Y., & Luo, W. (2019). DeepChain: Auditable and privacy-preserving deep learning with blockchain-based incentive. IEEE Transactions on Dependable and Secure Computing, 18(5), 2438–2455.",
  "Yan, Y., Kawahara, J., & Hamarneh, G. (2019). Melanoma recognition via visual attention. In Information Processing in Medical Imaging (pp. 793–804). Springer.",
  "Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. In Proceedings of the 35th International Conference on Machine Learning (pp. 5650–5659). PMLR.",
  "Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-IID data (arXiv:1806.00582). arXiv.",
];
refs.forEach(r => body.push(refEntry(r)));

// ---------- Document assembly ----------
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullet-list",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                 style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      },
      children: [titlePara, authorTableRow1, authorRowSpacer, authorTableRow2, spacer],
    },
    {
      properties: {
        page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
        column: { count: 2, space: 360 },
        type: SectionType.CONTINUOUS,
      },
      children: [abstractPara, indexTerms, ...body],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(`${ASSETS}/MedChainAI_ResearchPaper.docx`, buf);
  console.log("written");
});
