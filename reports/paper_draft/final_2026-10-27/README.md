# Final two-page Project 2 manuscript (27 October 2026)

**Research title:** *Whose Words Count? Reading Between the Samples in Historical isiXhosa NLP*.

This is the latest edited **research-body manuscript**. It restores explicit **RQ1** (controlled allocation versus page selection and held-out bigram loss) and **RQ2** (historical weighting estimands and permutation calibration). It retains the verified results, cultural context, author–year citations and original CCN style.

- `report.tex` — complete two-page research text with references afterwards
- `paper_references.bib` — checked bibliography
- `make_figures.py` — two reproducible, vector scientific plots with revised label positioning
- The inherited original course `ccn.sty` and `apacite.bst` are in `reports/paper_draft/`

## Compile locally

From this folder:
```bash
cp ../ccn.sty ../apacite.bst .
python -m pip install matplotlib numpy
python make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex8 report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdfinfo report.pdf | grep Pages
```

Expected: **three pages total**; two research pages including figures and equation, references on page three. The repository contains the editable source; the submitted PDF should be generated and visually inspected using these exact files. The source figure numbers are traceable to frozen evidence under commit `aff05ae34597f28309dc94561adbcd962901e2f4`.

**Submission warning:** This is not the complete administrative submission. The original complete NeurIPS 2024 questionnaire must be answered by the authors and appended after references; the Faculty of Science AI ethics disclosure and true contribution statements must be approved. Do not submit an unreviewed declaration. No experimental data or frozen result files were modified by this editorial revision.


## 27 October cultural-alignment and evidence update

The paper now explicitly treats archived **text exposure** as the common measurement problem: it asks how between-view allocation and within-view page selection affect Imvo tokenization (RQ1), and whether historical pattern changes depend on weighting publication units (RQ2). The quantitative source-specific context is 22.64% versus 1.40% EN-assigned *average-page* exposure for Imvo versus Isigidimi (29.47% versus 1.28% by eligible-word mass) over the overlap years. These are **archive-assigned text views**, not validated speaker-language identities.

New supplementary structural check (not a new causal experiment): `analysis/verify_page_structure.py` independently parses the author-provided original archive ZIP, checks its SHA-256 and reports XH-eligible four-gram positions by year and page position. The derived non-textual table `analysis/page_position_eligible_positions.csv` records the Imvo page-2/page-1 ratios 4.24 (1885), 3.69 (1888), compared with Isigidimi largest/smallest page-position means of 1.30 and 1.42. These are **post-hoc descriptive results**, not findings about editorials, author identities or political themes. The original source ZIP is not redistributed.

Source files `report.tex` and `make_figures.py` now match the compiled, visually reviewed research PDF made available with the 27 October 2026 editorial package. All prior experimental results and frozen evidence are unchanged.
