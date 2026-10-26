# Excellence Sprint Phase 3 - scientific decision

**Status: PHASE 3 PASSED, 2026-10-24; independent numerical verification PASS.**

## Objective and design

Continue the Phase 1 preregistered two-factor sampling intervention, retaining the
40 frozen Phase 2 tokenizer artifacts (2 model families x 4 sampler cells x 5
seeds) and the exact, previously selected train/validation split. Factor A is
the archive-view allocation (page-like P versus word-like W). Factor B changes
which eligible page is chosen within an archive view (page-conditional P versus
word-conditional W). PP/PW/WP/WW are all evaluated on **identical** held-out
page-view text from Imvo 1888 and 1889-1894; models were not refitted or tuned.

This is a **conditional prospective follow-up** to previous studies which have
already examined 1888 onward. The new factorial protocol was frozen before
these specific scores, but the material is NOT pristine independent held-out
evidence at the level of the overall research programme.

Metric: `100 * SentencePiece pieces / alphabetic characters`, pooled at the
source-view level. Bootstrap: 5,000 paired whole-publication-issue resamples,
seeds treated as five fixed training replications (not five new languages).

## Main decision table

| Frozen gate | 1888 effect (95% issue-bootstrap interval) | 1889-1894 | Outcome |
|---|---:|---:|---|
| F1 BPE English, view allocation | -1.878 [-1.972, -1.782] | -1.517 | SUPPORTED |
| F2 Unigram English, view allocation | -1.795 [-1.908, -1.671] | -1.368 | SUPPORTED |
| F3 BPE isiXhosa, view allocation | +0.453 [+0.407, +0.497] | +0.351 | SUPPORTED |
| F4 Unigram English, within-view selection | -0.431 [-0.705, -0.123] | -0.157 | SUPPORTED under frozen aggregate criteria |

Negative signs mean word-like sampling produces *fewer* pieces, i.e. more
compact representations; positive signs mean the opposite. F1's precommitted
magnitude floor is -0.30 pieces/100 characters; these are tokenizer-level
functional changes, **not social-harm estimates**.

The exact F1-F4 acceptance decisions are in `outputs/hypothesis_decisions.csv`.
All 32 family x time-period x archive-view x contrast rows are preserved in
`outputs/factorial_contrasts.csv`; unsuccessful or nonprimary comparisons were
not discarded.

## Broader evidence and qualifications

* The view-allocation intervention changes English training-character exposure
  from 18.46% (PP) / 18.64% (PW) to 26.27% (WP) / 26.58% (WW) on average.
  This distinction is caused by a sampling policy, not by randomized language.
* On 1888, BPE allocation shifts English segmentation by -1.878 but shifts
  isiXhosa segmentation +0.453 (opposite directions). Under Unigram, the
  analogous isiXhosa allocation effect is +0.555, an auxiliary observation.
* The F4 Unigram within-view result has heterogeneous seeds: four of five
  1888 effects are negative, and three of five 1889-94 effects are negative.
  Even though its **aggregate** frozen gate passes, do NOT describe it as
  universally seed-stable or predict a universal within-view effect.
* The within-view effect on BPE English 1888 is +0.136 with an interval that
  includes zero. This is a counterexample to a blanket claim that within-view
  page weighting helps both tokenizer families.
* Six-year-cluster 1889-94 sensitivity is available for every contrast, as a
  separate analysis based on just six independent year clusters; it does not
  replace publication-issue uncertainty.
* Testing 1888 and 1889-94 on the same newspaper means this is temporal
  generalization, NOT independent-title, modern-language, or cultural transfer.
* The paper's human newspaper-genre metadata were not available; no genre or
  political interpretation is warranted. The archive's `EnTxt`/`XhTxt` views
  are provided block classifications, not independently verified linguistic
  ground truth. Lower tokenization cost is not synonymous with lower harm.

## Integrity and independent reproduction

- Original newspaper language ZIP SHA256 and Phase 1 protocol SHA256 matched.
- All 40 Phase 2 model fingerprints and fixed 2,000-piece vocabulary sizes matched.
- 2,493 eligible page-view excerpts from 358 Imvo publication issues.
- 99,720 scores recomputed independently from original archive text;
  **zero piece-count disagreement**, zero unknown tokens.
- The independent implementation reconstructed the paired issue-level 5,000
  draw bootstrap: maximum absolute point-estimate residual 2.84e-15;
  maximum interval residual 4.17e-15.
- First run had only a CSV header/schema mismatch for six-year sensitivity
  columns. This was fixed and documented before hypothesis-table export;
  no model changes or test-driven tuning occurred.

## Next gated work, not completed

**Phase 4:** rerun PP and WW with repeated training text excluded under matched
page-view support, using TRAINING-only cross-issue exact 13-word shingles; then
evaluate the frozen F6 on the same held-out page-view set. Inspect train/test
text overlap as a contamination sensitivity without changing the test set.

**Phase 5:** apply the validation-locked smoothing `alpha=100` to a bigram
model with identical raw LM training strings and estimate BPC on the same
held-out strings. Apply F5; preserve a negative result if the advantage fails.

**Phase 6:** prepare the paper only when explicitly authorized; complete all
required AI ethics forms, contribution statement, code and exact page limit.
