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
