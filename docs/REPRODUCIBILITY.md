# Reproducing Project 2

The repository supports **two different operations**: checking the archived scientific outputs, and re-running model fitting from the source corpus. Do not describe a saved-result check as fresh retraining.

## 1. Environment

Use Python **3.13** and dependencies pinned in [`requirements.txt`](../requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository's historical archive and scripts were validated with the specific package versions in that file.

## 2. Fast verification: no retraining

Run from the repository root:

```bash
python -m unittest discover -s src -p 'test_*.py' -v
python scripts/check_repository.py
python src/run.py check
python src/audit/independent_ci.py --root .
python src/analysis/metric_relationships.py --root . --verify
```

The original-data-independent checker re-derives **84 saved contrasts and bootstrap intervals** from archived raw CSV rows. The code also has integrity tests for saved models and precommitted results.

## 3. Verify ArchiveLens from original archive text

Confirm the source ZIP hash shown in [`data/README.md`](../data/README.md) before using it:

```bash
sha256sum data/TxtLangFiles1870-1894_TB.zip
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
python src/archivelens/verify_independent.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify
python src/archivelens/verify_followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .
```

These compare independently reconstructed estimands and permutation statistics against archived results. They do not re-estimate cultural meaning or independently adjudicate the language labels.

## 4. Fresh model training and evaluation

This is the costlier end-to-end pipeline. It requires a verified copy of the original course-provided language-view archive and writes to a **new directory outside the checkout**:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay --dry-run
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

The recorded 26 October isolated reproduction used all original phase scripts, retrained **40 + 20 = 60** tokenizers, and compared regenerated outputs with the frozen records. See [audit](../reports/fresh_reproduction_2026-10-26.md). GitHub Actions runs verification, **not** the full training job.

## 5. Build the manuscript

```bash
cd paper
make pdf
```

`make pdf` requires LaTeX tools in addition to the pinned Python packages. Inspect the generated PDF's two-page research body visually and supply completed course declarations separately. The editable manuscript lives in [`paper/`](../paper/).

For the exact definitions, controls and limitations, read [methods](../reports/methods.md), [frozen Phase 1 protocol](../reports/protocol_phase1.md) and [paper evidence guide](../reports/PAPER_TO_EVIDENCE_GUIDE.md).
