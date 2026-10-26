# Project 2 Excellence Sprint / Phase 2 (research Phase 35)

**State:** TRAINING AND VALIDATION COMPLETE; PHASE 3 HELD-OUT SCORES NOT INSPECTED.
**Date:** 2026-10-24 (Africa/Johannesburg). The original Phase 33 results and Phase 1 protocol are immutable.

## Main scientific purpose

Complete the four-arm 2x2 sampling experiment (PP/PW/WP/WW) with five fixed seeds (441-445) and both BPE and Unigram vocabularies, using **only** the Imvo 1885 and Jan-Sep 1887 training texts and Oct-Dec 1887 duplicate-aware validation. Select a single token-bigram smoothing parameter for all comparisons. The preregistered F1-F6 hypotheses will be evaluated only in subsequent phases.

## Reproduction

Requires Python >=3.10, `pip install -r Phase2/requirements.txt`, and the original supplied `TxtLangFiles1870-1894_TB.zip` at a known path. The source newspaper text is not redistributed.

From the root of this package:

```bash
python Phase2/scripts/run_phase2.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --output Phase2/outputs
python Phase2/scripts/verify_phase2.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --root .
python Phase2/scripts/fresh_replay.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --root .
```

Replaying existing outputs rechecks every sampling input hash; to refit *all* model bytes in a scratch environment, copy the package and delete Phase2/outputs/{models,model_rows,model_training_manifest.csv,validation_scores.csv,validation_model_view_summary.csv,alpha_selection.csv,PHASE2_SELECTION_FREEZE.json,PHASE2_INDEPENDENT_VERIFICATION.json,PHASE2_FRESH_REPLAY.json}, then rerun. On different filesystem paths, serialized model hashes may change while the model vocabulary and encodings reproduce; the independent replay tests the latter.

## Frozen study configuration

- Source file SHA256: `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`.
- Phase 1 protocol SHA256: `bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4`.
- 342 training pages/86 issues; 48 validation pages/12 issues; no overlap.
- Training tokenizer budget: 500,000 +/- 30 lowercase ASCII alphabetic characters per fit; five seeds; BPE and Unigram; vocab 2,000; four sampler arms.
- Token bigram LM uses identical 545 raw training page-view excerpts for every model, not samples selected by arm; alpha grid {1,10,100}.
- Validation uses 344 duplicate-free 32-word windows, up to five per eligible page/view, rejecting every 13-word shingle already in same-view train text. The 344 windows comprise 104 English-view and 240 isiXhosa-view excerpts. Archive labels are block view assignments, **not verified language ground truth**.
- Global alpha chosen by lowest equal-view average validation BPC across all arms, families and seeds; no per-arm hyperparameter selection.

## Saved files

- `outputs/models/*.model|vocab`: the 40 frozen tokenizers.
- `outputs/model_training_manifest.csv`: seed, arm, model hash, sample hash, exposure and LM input counts.
- `outputs/validation_windows_manifest.csv`: window provenance and text hash, without newspaper text.
- `outputs/validation_scores.csv`: 13,760 per-window validation scores; bits at each permitted alpha.
- `outputs/validation_model_view_summary.csv`, `outputs/alpha_selection.csv`: validation-only aggregation.
- `outputs/PHASE2_SELECTION_FREEZE.json`: frozen alpha=100.
- `outputs/PHASE2_INDEPENDENT_VERIFICATION.json`: source-grounded complete scoring verification.
- `outputs/PHASE2_FRESH_REPLAY.json`: 4 fresh refits and 1,376 independent window-encoding comparisons.
- `PHASE2_GATE.md`, `RESEARCH_DECISION.md`, `PHASE3_START_PROMPT.md`: research decision and continuation.

## Scientific boundary

There are **no** new 1888-1894 effect sizes, fairness claims, cultural changes, or F1-F6 hypothesis outcomes here. The 1888+ text was previously examined in other phases, so later validation is a prospective *conditional follow-up*, not fresh external evidence. Repeated-text removal is the Phase 4 experiment; do not imply those filtered tokenizer fits have already been run.
