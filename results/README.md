# Results and scientific artifacts

This directory contains **frozen reported evidence**, not a folder of disposable training outputs. Do not rename or regenerate published files in place just to simplify the layout.

| Location | Contents |
| --- | --- |
| [`segmentation.csv`](segmentation.csv) | Main factorial segmentation estimates and uncertainty |
| [`bigram.csv`](bigram.csv) | Count-bigram bits-per-character contrasts |
| [`repetition.csv`](repetition.csv) | Matched-support and repeated-text controls |
| [`raw/`](raw/) | Per-excerpt scores and manifests used by independent checkers |
| [`models/`](models/) | Saved 60 SentencePiece models/vocabularies |
| [`archivelens/`](archivelens/) | Pattern candidates, estimands, calibrated warnings and explorer |
| [`figures/`](figures/) | Source study and ArchiveLens graphics |
| [`earlier/`](earlier/) | Intermediate historical development evidence |
| [`historical/`](historical/) | Separate historical-inference programme |
| [`fresh_reproduction_2026-10-26.json`](fresh_reproduction_2026-10-26.json) | External clean-retraining certificate |
| [`independent_audit_2026-10-26.json`](independent_audit_2026-10-26.json) | Independent numerical audit certificate |

Use [`python src/run.py check`](../src/run.py) or [`src/audit/independent_ci.py`](../src/audit/independent_ci.py) to inspect saved results without retraining. For new experiments, write output to a separate directory and record its configuration, software version, source hash and seed.

Source-to-paper mapping: [paper-to-evidence guide](../reports/PAPER_TO_EVIDENCE_GUIDE.md). Reproducibility instructions: [docs](../docs/REPRODUCIBILITY.md).
