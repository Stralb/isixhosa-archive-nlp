# Paper

The latest Project 2 manuscript is [`report.tex`](report.tex), revised 27 October 2026. The [26 October draft](../reports/paper_draft/) is kept for reference; its PDF is not the latest version.

To build the paper, first install the packages in [`requirements.txt`](../requirements.txt). You also need a LaTeX installation with `pdflatex`, `bibtex8` and the packages used by the original CCN template.

```bash
cd paper
make pdf
```

The build generates figures and `paper/report.pdf`. Check the PDF layout before submitting it. The current manuscript source does not include the completed ethics and contribution declarations.

The [NeurIPS questionnaire](neurips_2024_AUTHOR_REVIEW.tex) and files in [`author_review/`](author_review/) are drafts for the authors to complete and approve. They are not final submission forms.

Results and analysis: [paper evidence](../reports/PAPER_TO_EVIDENCE_GUIDE.md).
