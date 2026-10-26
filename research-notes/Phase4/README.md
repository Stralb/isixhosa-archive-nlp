# Excellence Sprint Phase 4 — executable reprint falsification

## Requirements and provenance

Use Python 3.13, `numpy`, `pandas`, `sentencepiece`, and `matplotlib` (the package versions in the Phase 2/3 requirements). Requires the ORIGINAL `TxtLangFiles1870-1894_TB.zip` supplied to this project (not redistributed here). Check source SHA256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`. Do not substitute a different archive edition. The whole Phase 1-4 ZIP must retain the `isiXhosa_Excellence_Sprint/` directory hierarchy.

## Reproduction

From any shell:

```
ROOT=/path/to/isiXhosa_Excellence_Sprint
SOURCE=/path/to/TxtLangFiles1870-1894_TB.zip
python "$ROOT/Phase4/scripts/train_phase4.py" --archive "$SOURCE" --output "$ROOT/Phase4/outputs"
python "$ROOT/Phase4/scripts/evaluate_phase4.py" --archive "$SOURCE" --output "$ROOT/Phase4/outputs"
python "$ROOT/Phase4/scripts/verify_phase4.py" --archive "$SOURCE" --root "$ROOT" --output "$ROOT/Phase4/outputs"
python "$ROOT/Phase4/scripts/plot_phase4.py" --output "$ROOT/Phase4/outputs"
```

`train_phase4.py` replays source-only 500,000-character draws; existing model outputs are reused only after hash comparison. To rerun model fitting from scratch, choose a new, empty output directory. `evaluate_phase4.py` checks Phase 3 text hashes and locks before scoring, and `verify_phase4.py` independently reconstructs 20 training samples, all 2,493 test page/views, 49,860 encodings, train-test exact overlap, and 20 bootstrap contrasts. A fresh retraining in a different directory produced all 20 identical learned `.vocab` files and exact evaluation CSVs.

## File inventory

- `PRECOMMIT.md`: frozen F6 design, controls, metric, decision and CI rule.
- `TECHNICAL_EXECUTION_NOTE.md`: logged manifest/writer and verifier ordering fixes; both output-only.
- `scripts/train_phase4.py`, `evaluate_phase4.py`, `verify_phase4.py`, `plot_phase4.py`: full execution.
- `outputs/support_audit.json`, `support_manifest.csv`: training-only source population and attrition.
- `outputs/models/*.model`, `*.vocab` and `model_rows/*.json`: 20 fixed tokenizer fits and sample hashes. No original newspaper text included.
- `outputs/TRAINING_LOCK.json`, `model_training_manifest.csv`: immutable replay information.
- `outputs/heldout_scores.csv`: all model-by-page/view token counts.
- `outputs/f6_contrasts.csv`, `f6_seed_effects.csv`, `f6_support_metrics.csv`, `phase3_original_reference.csv`: estimates, paired bootstrap, seed and prior-contrast details.
- `outputs/train_test_overlap_diagnostic.csv`: retained test-shingle overlap sensitivity, not a tuning filter.
- `outputs/PHASE4_DECISION.json`, `PHASE4_INDEPENDENT_VERIFICATION.json`, `PHASE4_FRESH_RETRAIN_CERTIFICATE.json`: verdict and checks.
- `outputs/phase4_en_repetition_robustness.png` and `phase4_xh_repetition_robustness.png`: reproducible illustrations.
- `PHASE4_GATE.md`, `RESEARCH_DECISION.md`, `PHASE5_START_PROMPT.md`: research status and next scientific task.

## Boundaries

Phase 4 is a frozen follow-up, not untouched external confirmation. Both control conditions use only 533 common training page/view cells; 62 original page/view cells are omitted because they have no eligible clean windows. `EnTxt`/`XhTxt` are the archive's block-level views, not language annotations. This study claims no human historical sentiment, culture, social-harm reduction, neural downstream benefit or improvement in NLP accuracy.
