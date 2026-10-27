# Current manuscript: *Whose Words Count?*

**Project 2 research-body source, revised 27 October 2026.**

This is the **current** CCN-formatted manuscript source. The earlier 26 October compiled draft in [`reports/paper_draft/`](../reports/paper_draft/) is retained for provenance, but should not be mistaken for the current submission.

## Files

| File | Purpose |
| --- | --- |
| [`report.tex`](report.tex) | Two-page scientific body and bibliography declaration |
| [`paper_references.bib`](paper_references.bib) | Research citations |
| [`make_figures.py`](make_figures.py) | Recreate the two vector figures from checked numerical evidence |
| [`neurips_2024_AUTHOR_REVIEW.tex`](neurips_2024_AUTHOR_REVIEW.tex) | Complete 15-question **draft**, requiring author approval |
| [`author_review/`](author_review/) | Unapproved declaration/contribution/submission worksheets |
| [`analysis/`](analysis/) | Post-hoc page-position summary and verification script |

The unchanged conference style files are kept in [`reports/paper_draft/`](../reports/paper_draft/). `make pdf` copies those dependencies locally; it does **not** change the course template.

## Build the paper

From the repository root, install Python 3.13 and the pinned dependencies. A TeX installation with `pdflatex`, `bibtex8`, `pslatex`, `apacite`, `microtype`, `hyperref`, `caption` and `needspace` is also required.

```bash
python -m pip install -r requirements.txt
cd paper
make pdf
pdfinfo report.pdf | grep Pages
```

The expected research manuscript has **two pages of body and figures, with references beginning on page three**. Verify the compiled layout visually rather than relying only on page count. The repository currently stores the **editable source**, not an approved assembled submission PDF.

To clean generated auxiliary files and copied styles:

```bash
make clean
```

## Before academic submission

The [submission-status worksheet](author_review/submission_status.md), [AI-use guidance](author_review/ai_use_disclosure.md), [contribution worksheet](author_review/contributions.md) and NeurIPS questionnaire are **author-review drafts**, not signed or finally approved declarations. Do not automatically approve proposed ethics answers or upload signed documents to a public repository.

The two-page body and references are not a replacement for the course-required completed NeurIPS questionnaire, Faculty AI ethics statement and confirmed individual contributions.

For scientific evidence and limitations, see the [paper-to-evidence guide](../reports/PAPER_TO_EVIDENCE_GUIDE.md). For the experimental implementation, see the [code map](../src/README.md).
