# Phase 1 gate decision: PASS

**Audit date:** 24 October 2026. **Scope:** sprint Phase 1 (research continuation Phase 34), not a claim that factorial experiments or publication are complete.

## New checks actually executed

1. Original Phase 33 `verify_phase33.py` rerun against the real archive, verified all 15 BPE model hashes, 37,395 scored rows, 194 independent source-text re-encodings and 60 aggregate recomputations: PASS.
2. Independent prospective split audit: **342 train pages / 86 issues**, **48 validation pages / 12 issues**, **zero train/validation issue overlaps**; future-year text was not accessed by the **new factorial source audit**. The separate prior-result re-verifier re-reads historical 1888+ excerpts that were already scored in Phase 33; no new factorial outcome was scored.
3. Four-factor probability algebra verified: `PP` / `PW` / `WW` correspond exactly to the Phase 33 P / M / W sampling distributions under equivalent eligibility; `WP` completes the crossed design. Sum of probability weights for every arm is 1.
4. Real source train-only repeated-text screen: **16,691** distinct repeated 13-word shingle types across different training issues (from 684,318 shingle types total). Duplicate-free windows remain available for **335/341** eligible isiXhosa-view page-views and **198/254** English-view page-views.
5. Train/validation contamination audit: **12.85%** of validation isiXhosa-view 13-grams and **1.24%** of English-view 13-grams occur in training. The new protocol excludes training-overlapping windows from validation-only tuning.
6. Four-arm sampler smoke-tested under three train-only input variants (ordinary, duplicate-free, common-support unfiltered): **12/12 deterministic checks pass**, including exact replay of the same random streams and character budgets.
7. Independent verifier re-derived source counts, both view-allocation probabilities, every arm's probability normalization, sampled duplicate-window counts and train/validation overlap. See `outputs/verification.json` for individual assertions and frozen-protocol hash.

## Feasibility findings (not new held-out results)

- The train-only expected **word-draw allocation** is **23.55%** English-view under page-like selection versus **32.56%** under word-like selection (a 9.01-point difference). These are expected **draw probabilities**, NOT measured final alphabetic-character exposures or held-out performance; final exposure is tested in Phase 2.
- The 1887 Q4 validation cohort is small but has **28,343** English-view and **58,710** isiXhosa-view candidate duplicate-free 32-word windows (overlapping windows, not independent samples).
- The repeated-text filter removes all eligible windows from 56 of 254 English page-views. Phase 4 therefore requires the precommitted **common-support comparator**; otherwise removing repeated material would be confounded with dropping particular pages.

## Gates NOT yet passed

- No new 2x2 tokenizer has been fitted as a research result; Phase 2 will train and validate.
- No new 1888+ factorial test metrics have been evaluated; Phase 3 must remain locked until validation ends.
- No new LM causal-consequence, historical sentiment, genre, social-harm or cultural-change result exists.
- No main extended abstract, contribution statement or ethics questionnaires have been completed. Submission gates remain critical.

**Decision:** Phase 1 PASSED. Proceed to Phase 2 using `FROZEN_PROTOCOL.md` and its SHA256 in `outputs/verification.json`. Do not modify its hypotheses after evaluating held-out text; if a technical issue forces a correction, save a dated amendment before evaluation.
