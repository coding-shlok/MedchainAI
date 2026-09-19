# Research Paper Generation

## Files in This Folder

- **MedChainAI_ResearchPaper.docx** — Final IEEE-format paper (two-column layout, APA references)
- **gen_docx.js** — Node.js script that generates the .docx file from scratch using the docx library
- **gen_figures.py** — Python/Matplotlib script that generates all 4 figures from logged metrics
- **gen_explainer.js** — One-page project explainer (for interviews/quick pitches)
- **fig1_pipeline.png** through **fig4_chain_architecture.png** — Embedded figures

## How to Regenerate

### All Figures

Requires training results to exist (metrics.json files in results/ directory):

```bash
cd research_paper
python gen_figures.py
```

This generates fig1–4 from:
- `results/isic2019_binary_efficientnet_b0/` — training metrics
- `results/attack_sign_flip/metrics.json` — poisoning defence evaluation

### Full Paper

```bash
cd research_paper
node gen_docx.js
```

Output: MedChainAI_ResearchPaper.docx (placed in the same directory)

### Project Explainer (Interview Brief)

```bash
cd research_paper
node gen_explainer.js
```

Output: MedChainAI_Project_Explainer.docx

## Paper Structure

- **Title & Authors:** 5 authors, all from Mukesh Patel School (SVKM's NMIMS)
- **Abstract:** ~300 words, emphasizing blockchain as co-equal design objective
- **Sections I–IX:**
  - I. Introduction
  - II. Literature Review (+ II-A: Research Gap and Novelty)
  - III. Problem Statement
  - IV. Datasets (A: Binary framing, B: Train/test split)
  - V. Pre-processing (Dirichlet partitioning)
  - VI. Proposed System Architecture (A–F: backbones, what gets federated, ledger, model training, reputation/Byzantine defence, independent audit)
  - VII. Experimental Setup and Evaluation Metrics
  - VIII. Results and Analysis (A–F: classification performance, poisoning detection, ledger provenance, current ViT status, discussion, limitations)
  - IX. Conclusion
- **References:** 28 peer-reviewed papers in APA 7th edition format (alphabetical, hanging-indent)

## Key Numbers in Paper

- **ISIC 2019:** 25,331 images, 8 classes, 20,265 training / 5,066 held-out
- **Binary task:** Malignant (MEL, BCC, SCC) vs. Benign (all others)
- **Federated:** 5 hospitals, Dirichlet(α=0.5), 20–50 rounds
- **EfficientNet-B0:** 80.6% accuracy, 0.778 macro-F1 (20 rounds)
- **ResNet-18 (8-class):** 71.5% accuracy
- **Poisoning detection:** 100% cumulative recall (3 of 10 clients malicious)
- **Gas costs:** submitUpdate ~188k gas (most frequent call)

## Recent Edits (Latest Session)

1. **References:** Converted from numbered [1]–[27] to proper APA format with real papers
   - Used actual papers (McMahan et al. 2017, Blanchard et al. 2017, Esteva et al. 2017, Tschandl et al. 2018, etc.)
   - All in-text citations updated from [n] to (Author, Year) format

2. **Humanization:** Reduced AI-generated tone
   - Trimmed ~30 em-dashes (kept ~30 where they add clarity)
   - Reduced "rather than" constructions
   - Removed redundant emphatic words ("deliberate," "genuine," "authentic")
   - Varied sentence rhythm and connectives

3. **Structure:** No changes to section order, results, or technical content
   - All blockchain content (VI-C, VI-F, VIII-C, Table IV) already present
   - Reputation mechanism already thoroughly covered
   - All 4 figures embedded and captions updated

## Before Your Interview

- Print or have MedChainAI_ResearchPaper.docx ready to share
- Know the headline numbers (80.6%, 100% detection recall, 188k gas)
- Be ready to explain the three gaps your work closes
- Have the project explainer (one-pager) as a fallback if interviewer prefers a brief
- Practice the 60-second pitch: federated learning + Byzantine defence + blockchain audit trail on real medical data

