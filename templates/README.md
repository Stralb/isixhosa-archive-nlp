# Official NLP 2026 templates (preserved exactly)

The **full original user-supplied ZIPs** are tracked in [original/](original/). The source archives were uploaded using Git objects with byte-for-byte verification; extracted files in this directory were separately checked against their corresponding ZIP members in a successful GitHub Actions job.

| Original file | SHA-256 | Extracted source folder |
| --- | --- | --- |
| `_NLP_2026__Extended_Abstract_Template.zip` | `6fcc74188f418b9e6e35fa76837c34abd5b065b78c8287f2a3ea606026af5f55` | [`CCN_2026/`](CCN_2026/) |
| `NeurIPS_2024_checklist.zip` | `9e21a868b448b31f978e90c21e8aa2b61e766b11b94fc907cae13bf0c9871e66` | [`NeurIPS_2024/`](NeurIPS_2024/) |

The extracted **CCN 2026 extended abstract** source retains `ccn_style.tex`, `ccn_style.bib`, `ccn.sty`, `apacite.bst` and the original 2.35 MB illustration `images/image.jpeg`.

The extracted **NeurIPS 2024** package retains the original `neurips_2024.tex`, `neurips_2024.sty` and `neurips_2024.pdf`. These are intact original materials, **not** the authored submission.

## Our drafted submission

Use [`reports/paper_draft/`](../reports/paper_draft/) for the full drafted, scientifically argued extended abstract. It is built in the **original CCN style** from these template files. We removed the two forced dummy page breaks from the example; they are demonstrations, not required blank pages.

- [`report.pdf`](../reports/paper_draft/report.pdf) — compiled draft, 9 pages: two-page scientific body, then references, completed-to-review NeurIPS checklist and author-only declaration/contribution placeholders
- [`report.tex`](../reports/paper_draft/report.tex) — the full LaTeX source; [bibliography](../reports/paper_draft/paper_references.bib), [checklist source](../reports/paper_draft/neurips_checklist_filled.tex), [figure](../reports/paper_draft/figures/main_evidence.pdf)
- [`STORYLINE_AND_FINALISATION.md`](../reports/paper_draft/STORYLINE_AND_FINALISATION.md) — rubric-aware scientific argument, all source claims and required final author actions
- [Source ZIP](../reports/paper_draft_bundle_2026-10-26.zip) — complete editable manuscript and preview.

**Required before submission:** confirm real registered authors/contributions and all provisional NeurIPS answers; add the exact Faculty of Science AI Ethics form. The archive's public redistribution rights should also be confirmed.

To rebuild the paper from `reports/paper_draft/` with a TeXLive distribution providing `pdflatex` and `bibtex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

**Two-page hard cap**: the PDF's *scientific body*, including the figure, ends on page 2; references start on page 3. See the assignment brief for the ordering of checklist and ethics declarations. The source page numbers in the original CCN *example* are not the submission page budget.
