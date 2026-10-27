> **Archived 26 October 2026 paper draft.** The current manuscript source and status are in [`paper/`](../../paper/). Keep this folder for provenance and original CCN style dependencies; do not submit the older PDF as the 27 October version.

# Two-page Responsible NLP extended abstract (revised scientific draft)

**Authors, in requested order:** Arno Strauss (2613224), Priyanka Gohil (2586288), Mohammed Bangie (2610990).

This directory contains the actual source for the paper draft, built with the **original course CCN template style** (`ccn.sty`, `apacite.bst`), references, source code for the integrated three-panel figure, and the completed 9-page PDF bundle. The **scientific body ends on page 2**, with references beginning on page 3. Remaining pages include an author-review NeurIPS checklist, provisional Faculty AI disclosure and contribution statement placeholders.

**Important:** this is **not yet submission-ready**. The group must fill the official Faculty AI Ethics Statement and actual individual contributions, review the NeurIPS 2024 checklist and confirm dataset redistribution permissions. The authors' order and identifiers are supplied by the user, but responsibilities were not supplied and must not be fabricated.

## Narrative

The paper tests a common question: how are scientific claims affected by the unit of archival sampling? It uses **two complementary but separate studies**:

1. Controlled training-exposure interventions on Imvo show English/isiXhosa segmentation trade-offs and different downstream bigram-model effects.
2. ArchiveLens measures sensitivity of historical *surface character-pattern frequency* claims to four alternative sampling estimands. Its own alert calibration and issue/repetition diagnostics distinguish observed sensitivity from unsupported cultural inference.

The proposed contribution is a reproducible *measurement-validity workflow* for historical-language NLP, not an original generic fairness metric, a new tokenizer algorithm, or validation of cultural/semantic change.

## Build locally

```bash
python -m pip install matplotlib numpy
python make_figure.py
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex8 report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdfinfo report.pdf | grep Pages
```

For clean reproduction, use an installation with `pslatex`, `apacite`, `microtype`, `hyperref`, and `caption` available. The original unmodified CCN and NeurIPS ZIPs are separately supplied in the **original_templates/** folder of the user-facing bundle. Do not silently replace the course conference style with generic `article` style.

## Files

- `report.tex` — revised two-page CCN scientific narrative, plus references and author-review declarations.
- `report.pdf` — compiled PDF containing two-page scientific body and declarations.
- `paper_references.bib` — related work.
- `make_figure.py` — a **reproducible integrated three-panel figure** with numeric inputs pinned to published result files.
- `figures/integrated_evidence.pdf` — vector graphic used in the paper.
- `figures/integrated_evidence.png` — image preview for web reading.
- `neurips_checklist_filled.tex` — complete NeurIPS 2024 questionnaire source with explicit author-review questions and rationale (not final approval).
- `ccn.sty` and `apacite.bst` — course-provided CCN template dependencies.

## Full research and evidence

<https://github.com/Stralb/isixhosa-archive-nlp/blob/main/reports/PAPER_TO_EVIDENCE_GUIDE.md>

This guide connects each result to the full original score tables, independent audits, ArchiveLens raw-output data, reproducible code, and limitations. The paper must still stand alone: GitHub gives depth, not extra body pages.
