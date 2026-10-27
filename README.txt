NLP Project 2 - Responsible NLP in historical isiXhosa newspapers
CURRENT MANUSCRIPT: paper/README.md
READER GUIDE: reports/PAPER_TO_EVIDENCE_GUIDE.md
REPOSITORY MAP: docs/README.md
Main research: sampling allocation, BPE/Unigram, bigram language modelling.
ArchiveLens: sampling-sensitive historical surface-pattern audit with calibrated warnings.

ENVIRONMENT
  Python 3.13
  python -m pip install -r requirements.txt

QUICK VERIFICATION (saved outputs; no retraining)
  python -m unittest discover -s src -p 'test_*.py' -v
  python src/run.py check
  python src/audit/independent_ci.py --root .
  python src/analysis/metric_relationships.py --root . --verify

ARCHIVELENS SOURCE-DATA VERIFICATION
  Original archive: data/TxtLangFiles1870-1894_TB.zip
  Check its SHA-256 against data/README.md before reproduction.
  python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
  python src/archivelens/verify_independent.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens
  python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify

FULL TRAINING REPRODUCTION (costlier than the checks above)
  python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay

IN-DEPTH DOCUMENTATION
  reports/PAPER_TO_EVIDENCE_GUIDE.md - paper claim -> files/results -> limitations
  reports/methods.md and reports/findings.md - main controlled study
  reports/ARCHIVELENS_RESULTS_2026-10-26.md - original ArchiveLens audit
  reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md - calibration and attribution
  src/archivelens/README.md and src/archivelens/FOLLOWUP.md - tool usage
  results/archivelens/explorer.html - offline interactive evidence explorer
  .github/workflows/ci.yml - reproducibility checks on GitHub Actions

Archive EnTxt/XhTxt blocks are not adjudicated word-level language labels.
Historical pattern changes are not validated cultural meaning or fairness harms.
The original two-page extended abstract and ethical declarations are separate
mandatory submission documents; GitHub source code is not a substitute.
