# Scientific argument and finalisation notes (NLP Project 2)

## High-mark research story

**Thesis.** Historical newspaper NLP inherits the archive's sampling and transcription choices. A responsible evaluation must test (i) how those choices redistribute computational representation across archive language views, and (ii) whether a quantitative claim about the text remains stable when the unit of measurement changes.

**Research question.** How do defensible corpus sampling choices alter learned tokenization outcomes and the stability of quantitative claims about historical isiXhosa newspaper text?

The argument has *two distinct empirical branches*, not one causal pathway:

1. **Interventional branch (Imvo).** Factorially vary language-view allocation and within-view page selection with a fixed 500,000-alphabetic-character tokenizer budget, BPE/Unigram, five seeds. Measure language-view-specific segmentation and smoothed count-bigram BPC on disjoint-date splits. Report the English improvement/isiXhosa segmentation cost and the lack of universal alignment between compactness and downstream BPC. The intervention is on sampling probabilities, *not* on historical language or speakers.
2. **Measurement-audit branch (Imvo and Isigidimi).** Introduce ArchiveLens and audit 120 *1887-selected* within-word character patterns for 1885-to-1888 frequency trends under four defensible estimands. Calibrate its warnings against month-stratified whole-issue permutations and Holm corrections. Explain influence with issue deletion, page position, and exact repeated-passage masks. Do not claim those text patterns are morphemes, cultural concepts or changes in historical ideology.

**Why the two branches belong in one paper:** they challenge the assumption that a pipeline decision is a neutral background choice. Branch 1 shows consequences for NLP model evaluation; Branch 2 shows consequences for a research measurement about the underlying text. The ArchiveLens analysis was not produced by BPE; *no mediator relationship is claimed*.

**Responsible NLP insight:** A reported positive result can be incomplete (English-only segmentation gain), fragile (sampling estimator changes sign) or statistically unremarkable (raw sign-reversal counts within a shuffled-year reference). A good research instrument should quantify and disclose each condition, including negative tests, before any culturally situated interpretation.

## Three results that must survive paper compression

1. **BPE 1888 allocation:** English-view pieces/100 alphabetic chars -1.87754, issue-bootstrap 95% CI [-1.97159,-1.78181]; isiXhosa-view +0.45259, CI [+0.40684,+0.49737]. Unigram shows the same opposite directions. These are archive-assigned *views*, not language-identity gold labels.
2. **Downstream BPC:** BPE English allocation -0.0372167 BPC (95% issue CI [-0.042364,-0.032119]); within-view selection +0.0193781 BPC. The whole WW vs PP policy does not show reliable benefit across 1889-94 under the six-year bootstrap, so isolated gains should not be generalized as policy optimality.
3. **ArchiveLens:** Imvo 23/120 and Isigidimi 4/120 sign reversal warnings; none has confidently opposite directions under both tested estimands. Under 4,999 whole-issue year permutations Imvo's count is *below* the null mean ~29 (upper-tail p=.8882). Nevertheless 29/120 Imvo pattern-specific estimator-gap magnitudes survive Holm correction (zero for Isigidimi). Removing cross-issue repeated 13-word passages makes 10 original Imvo sign-reversal alerts disappear, with the important caveat that the eligible sample changes.

## What distinguishes this from prior work?

- Existing papers establish multilingual token length disparities (Petrov et al., 2023), the weakness of tokenizer fertility as a downstream proxy (Ali et al., 2024), documentation of data population (Bender & Friedman, 2018), and archive collection biases (Beelen et al., 2025). **Do not claim to discover these general phenomena for the first time.**
- The coursework contribution is an evidence-rich, reproducible *archive-specific, two-branch measurement-validity case study* with a working auditor, statistical calibration, controls and a principled negative finding.
- The prior root-frequency analysis of these newspapers (Schoots, 2023) means four-character strings cannot be billed as new historical/cultural insights.

## Why this fits the marking rubric

- **Discussion and original insight (30%).** State where technical optimizations or apparent textual changes mislead; discuss null and sensitivity results explicitly. Distinguish computational disparities from morally meaningful harms. Specify what human/contextual validation is still missing.
- **Methodology (25%).** Controlled factorial design, date-group train/validation/test, fixed budgets, seeded tokenizers, issue bootstrap. ArchiveLens feature selection isolated to 1887; four estimands; clustered month/issue permutation and Holm correction; repeated-text attribution. Disclose prior inspection of test years.
- **Results (25%).** Use one two-panel figure, signed effects with confidence intervals, baseline sampling proportions, a downstream BPC contrast, and the calibrated ArchiveLens result including the negative count finding. Avoid dozens of secondary phase summaries.
- **Related work and structure (20%).** Cite archive provenance, fair-tokenization literature, dataset documentation and collection-bias work. Use one question and a clean progression, not a list of unrelated experiments.

## Read more in GitHub

- `reports/PAPER_TO_EVIDENCE_GUIDE.md`: paper claim -> exact evidence, methods and limitations.
- `reports/methods.md`, `reports/findings.md`, `results/segmentation.csv`, `results/bigram.csv`: original controlled experiment.
- `reports/ARCHIVELENS_RESULTS_2026-10-26.md`: the original claim audit.
- `reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md`: calibration, multiple testing, issue/repetition attribution.
- `results/archivelens/`, `src/archivelens/`: reproducible results, explorer and independent verification.

## Mandatory author-only finalisation before submission

- [ ] Replace `[Confirm all group authors]` in `report.tex` with exactly the registered project authors.
- [ ] Insert the official Wits Faculty of Science AI Ethics Statement / policy text. The draft contains a truthful *starter disclosure* but not the required signed or institution-specific form.
- [ ] Confirm and finalise the group contribution statement with all actual group members. Do not invent roles or percentages.
- [ ] Review every NeurIPS checklist answer (especially code of ethics, licensing, release safeguards and compute). The questionnaire is included with the original questions and guidelines preserved; this version is **not fully signed off**.
- [ ] Verify original archival text redistribution rights, since the source ZIP files are already public in GitHub. If permission is absent, remove or restrict them before final release.
- [ ] Confirm whether submission permits a direct GitHub URL in the paper and include the source-code repository as an accompanying deliverable.
- [ ] Compile `report.tex` from this directory and verify scientific body is **only pages 1-2**, with references beginning on page 3. Do not remove limitations to force an arbitrary high score.

## Build

```bash
cd reports/paper_draft
python make_figure.py
pdflatex -interaction=nonstopmode report.tex
bibtex8 report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

The original CCN source deliberately had two dummy page breaks before its sample abstract. This draft removes them while retaining `ccn.sty`, the 10-pt, letter-paper two-column geometry and the supplied apacite bibliography style. Full original ZIP archives are separately preserved under `templates/original/`.
