# Phase 5 — Downstream count-bigram language-model validation

**F5 SUPPORT** with a tightly bounded claim: archive language-view allocation improves English-view prediction with a frozen count-bigram LM, while page-within-view selection can neutralize that improvement. Exact main numbers in `RESEARCH_DECISION.md`. This phase uses no new tokenizer fitting, no neural models and no abstract.

## Required data

Maintain the `isiXhosa_Excellence_Sprint` relative folder tree (all Phase 1–4 assets must remain). The original user-supplied newspaper source `TxtLangFiles1870-1894_TB.zip` is required separately at an arbitrary path. Its SHA256 is `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`. It is **not** redistributed in the handoff, as prior phase packages also omitted it.

Tested with Python 3.13.5, numpy 2.3.5, pandas 2.2.3, sentencepiece 0.2.1, matplotlib 3.10.8. Install with `pip install -r Phase5/requirements.txt` in an appropriate environment.

## Reproduce all Phase 5 outputs

From the directory containing `isiXhosa_Excellence_Sprint/`, use absolute or correctly resolved paths:

```bash
ROOT=/path/to/isiXhosa_Excellence_Sprint
CORPUS=/path/to/TxtLangFiles1870-1894_TB.zip
OUT=$ROOT/Phase5/outputs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python $ROOT/Phase5/scripts/run_phase5.py --root $ROOT --archive "$CORPUS" --output "$OUT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python $ROOT/Phase5/scripts/verify_phase5.py --root $ROOT --archive "$CORPUS" --output "$OUT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python $ROOT/Phase5/scripts/secondary_sensitivity.py --output "$OUT"
python $ROOT/Phase5/scripts/plot_phase5.py --output "$OUT"
```

`run_phase5.py` is restart-safe: each finished model has its own row CSV and SHA256 audit. Use `--limit 10`, `--limit 20`, `--limit 30` for staged fitting/score evaluation if execution-time limits apply; a final call with no `--limit` builds the summary. With 40 frozen Phase 2 models and 2493 evaluation page-views, it must return **99720** score rows and 32 factorial contrasts.

For a **full cold replay** to a clean output folder, change `OUT` to a fresh path. `Phase5/PRECOMMIT.md` and the Phase 1 protocol are immutable; if their SHA256 changes the scorer refuses to run. `verify_phase5.py` runs a separate source parser and separate count bigram formula and re-derives every saved cost. It can checkpoint in batches with `--max-new 10`.

All results are saved in `outputs/`:

- `F5_DECISION.json`, `PHASE5_MANIFEST.json`, `PHASE5_INDEPENDENT_VERIFICATION.json` — frozen decisions and verification. `Phase5/PHASE5_FRESH_REPLAY_CERTIFICATE.json` documents the fresh extraction and 134 identical replay files.
- `heldout_lm_scores.csv` — 99720 scored example-model combinations, including raw bits and shared char denominators.
- `bpc_factorial_contrasts.csv` — all 32 BPC contrasts and 95% paired whole-issue confidence intervals; `per_seed_effects.csv`, `per_year_effects.csv`.
- `per_model_view_metrics.csv` — family x period x view x five seeds x four arms (160 rows), with pooled bits and alphabetic-character denominators.
- `heldout_repetition_overlap.csv`, `no_training_13gram_overlap_contrasts.csv`, `no_training_13gram_overlap_bootstrap.csv` — labelled source-excerpt overlap and post-selection secondary sensitivity.
- `phase5_language_allocation_bpc.png`, `phase5_policy_offset.png` — figures.
- `model_rows/*.csv` and `model_rows/*.audit.json` — 40 checkpointed raw score files and their model SHA256s; `independent_verification_models/*.json` — forty independent checks.

## Claim boundaries

The studied years 1888–94 were seen by researchers in earlier phases; independent numerical replay is NOT a fresh external dataset. `EnTxt`/`XhTxt` are archived language-view labels. No human genre, cultural sentiment, semantic task, neural LM or social fairness assessment is contained here. Phase 6 remains pending.
