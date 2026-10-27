NLP Project 2 - Historical isiXhosa newspaper NLP
University of the Witwatersrand

Read the project overview in README.md.
Latest paper source: paper/report.tex (build instructions: paper/README.md).
Methods and paper evidence: reports/PAPER_TO_EVIDENCE_GUIDE.md.

Python environment
  Python 3.13
  python -m pip install -r requirements.txt

Check the saved experiment results (no model retraining)
  python -m unittest discover -s src -p 'test_*.py' -v
  python scripts/check_repository.py
  python src/run.py check
  python src/audit/independent_ci.py --root .
  python src/analysis/metric_relationships.py --root . --verify

Check ArchiveLens against the original archive
  python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
  python src/archivelens/verify_independent.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens
  python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify
  python src/archivelens/verify_followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .

Rebuild the models (slower; use a new output directory)
  python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay

Paper (requires pdflatex, bibtex8 and the course LaTeX dependencies)
  cd paper
  make pdf

Data hashes and archive provenance: data/README.md
ArchiveLens usage: src/archivelens/README.md
Fresh retraining check: reports/fresh_reproduction_2026-10-26.md

The EnTxt and XhTxt fields are archive-assigned text views, not manually
verified language labels. The later test years were studied during earlier
exploration. Check the paper's limitations before interpreting results.
