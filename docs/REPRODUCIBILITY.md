# Reproducing the coursework experiments

## Source archive

Obtain the original `TxtLangFiles1870-1894_TB.zip` from your course distribution or the IsiXIT archive. The data are not mirrored by this repository. Compare its SHA256 to the pinned value in [`DATA_ACCESS_AND_ETHICS.md`](DATA_ACCESS_AND_ETHICS.md). If the source changes, numerical results may not match.

## Environment

The tested environment was Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SentencePiece 0.2.1 and Matplotlib 3.10.8. A CPU is sufficient. Set BLAS thread counts to one where necessary for consistency.

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Fast audit of saved results

From repository root:

```bash
sh coursework/isiXhosa_Excellence_Sprint/Phase6/submission_code/RUN_FAST_CHECK.sh /absolute/path/to/TxtLangFiles1870-1894_TB.zip
```

Expected summary:

```text
PASS 7 independent scientific tests; point effects independently reproduced: 32 + 32
SOURCE 375c5864... MODELS 40 SCORE_ROWS 99720 HELDOUT_PAGES 2493
FAST AUDIT AND ALL FOUR FIGURES: PASS
```

This is a numerical and artifact audit. It reads the existing frozen model outputs. It does not retrain forty tokenizers.

## Full rerun from source data

Enter the original research directory to preserve relative paths:

```bash
cd coursework/isiXhosa_Excellence_Sprint
SOURCE=/absolute/path/to/TxtLangFiles1870-1894_TB.zip
python Phase2/scripts/run_phase2.py --archive "$SOURCE" --output Phase2/outputs
python Phase2/scripts/verify_phase2.py --archive "$SOURCE" --root .
python Phase3/scripts/run_phase3.py --root . --archive "$SOURCE" --output Phase3/outputs
python Phase3/scripts/verify_phase3.py --root . --archive "$SOURCE" --output Phase3/outputs
python Phase4/scripts/train_phase4.py --archive "$SOURCE" --output Phase4/outputs
python Phase4/scripts/evaluate_phase4.py --archive "$SOURCE" --output Phase4/outputs
python Phase4/scripts/verify_phase4.py --archive "$SOURCE" --root . --output Phase4/outputs
python Phase5/scripts/run_phase5.py --root . --archive "$SOURCE" --output Phase5/outputs
python Phase5/scripts/verify_phase5.py --root . --archive "$SOURCE" --output Phase5/outputs
```

The complete, tested sequence is in [`coursework/isiXhosa_Excellence_Sprint/README.txt`](../coursework/isiXhosa_Excellence_Sprint/README.txt); follow it for restarts, the Phase 1 baseline dependency and exact output destinations. Fresh reruns should use a **copy** of the research directory. Do not overwrite frozen checkpoints.

## Verification evidence

- [`Phase6/PHASE6_FRESH_REPLAY_CERTIFICATE.json`](../coursework/isiXhosa_Excellence_Sprint/Phase6/PHASE6_FRESH_REPLAY_CERTIFICATE.json)
- [`Phase6/outputs/PHASE6_SCIENCE_AUDIT.json`](../coursework/isiXhosa_Excellence_Sprint/Phase6/outputs/PHASE6_SCIENCE_AUDIT.json)
- [`Phase6/CLAIM_LEDGER.md`](../coursework/isiXhosa_Excellence_Sprint/Phase6/CLAIM_LEDGER.md)
- [`Phase4/outputs/PHASE4_FRESH_RETRAIN_CERTIFICATE.json`](../coursework/isiXhosa_Excellence_Sprint/Phase4/outputs/PHASE4_FRESH_RETRAIN_CERTIFICATE.json)

The earlier exploratory studies and historical-inference programme have separate protocols. Their public code snapshots are provided as supplementary provenance and are **not part of the frozen final coursework factorial results**.