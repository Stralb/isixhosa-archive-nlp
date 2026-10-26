# Phase 3: reproducible frozen factorial evaluation

This folder contains 1888 and 1889-1894 held-out results from the already-trained
Phase 2 Imvo tokenizers. No newspaper text is redistributed. Read
`RESEARCH_DECISION.md` and `PHASE3_GATE.md` for claims and boundaries.

## Inputs

- `../Phase1/FROZEN_PROTOCOL.md`: SHA256 `bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4`.
- `../Phase2/outputs/models/`: 40 exact validated tokenizer artifacts.
- `../Phase2/outputs/PHASE2_SELECTION_FREEZE.json`: LM smoothing alpha 100 frozen.
- Original `TxtLangFiles1870-1894_TB.zip`: SHA256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`.
- Python 3.11+, NumPy 2.x, pandas 2.x, SentencePiece 0.2.x, matplotlib for figures.

## Replay

From the unpacked package directory whose child is `isiXhosa_Excellence_Sprint`,
run with your newspaper ZIP path substituted:

```bash
python isiXhosa_Excellence_Sprint/Phase3/scripts/run_phase3.py \
  --root isiXhosa_Excellence_Sprint \
  --archive /path/to/TxtLangFiles1870-1894_TB.zip \
  --output /tmp/phase3_replay
python isiXhosa_Excellence_Sprint/Phase3/scripts/verify_phase3.py \
  --root isiXhosa_Excellence_Sprint \
  --archive /path/to/TxtLangFiles1870-1894_TB.zip \
  --output /tmp/phase3_replay
python isiXhosa_Excellence_Sprint/Phase3/scripts/make_figures.py --output /tmp/phase3_replay
```

The independent verifier never relies on `run_phase3.py` for computing its
encoding or issue-bootstrap checks. It rescans the archive, checks all model
hashes and all saved scores, and recomputes estimates and intervals.

## Outputs

- `heldout_pageview_manifest.csv`: document/view source identifiers, length,
  issue, year, SHA256 of evaluated text (no redistributed text).
- `heldout_scores.csv`: 99,720 fixed-model piece counts, unknowns, paired rows.
- `per_arm_seed_metrics.csv`: corpus-character-weighted group metrics.
- `factorial_contrasts.csv`: all 32 primary/auxiliary contrasts, 5,000-draw
  publication issue CI, and six-year cluster sensitivity in later years.
- `per_seed_and_year_effects.csv`: source-seed and annual variability.
- `hypothesis_decisions.csv`: preregistered F1-F4 decisions, not cherry-picked.
- `PHASE3_INDEPENDENT_VERIFICATION.json`: complete numerical provenance.

CIs describe variability when resampling issues conditional on the fixed five
model seeds. Past work inspected later Imvo issues: Phase 3 is NOT an external
untouched replication. F5 BPC and F6 duplication falsification are NOT included.
