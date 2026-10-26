# Phase 4 research decision — Reprints are not the sole explanation

**Question:** Can repeated newspaper paragraphs alone explain why page-like versus word-like training changes English-view historical tokenizer segmentation?

**Hypothesis F6 (preregistered):** The English-view PP-WW segmentation advantage remains positive and practically >0.30 pieces/100 characters on 1888 after removing training-only cross-issue repeated 13-word strings from the sampled 32-word windows, with an issue-bootstrap 95% CI excluding zero and later 1889-94 results in the same direction.

**Result:** Supported by five of five model seeds on both evaluation periods. Filtered PP-WW effect +2.389 [2.182,2.590] (1888); +2.119 [2.050,2.185] (1889-94). Unfiltered models trained on *exactly the same page/view support* also show effects: +2.097 [1.905,2.280] and +1.615 [1.540,1.687]. For context, the earlier all-support Phase 3 PP-WW effects were +1.742 and +1.087 (not matched for source availability).

**Inference:** The sampling effect is not eliminated by removing this specific class of training newspaper repetition; the earlier effect is therefore not simply an artefact of keeping duplicate windows. Because 56 English and 6 isiXhosa eligible page/view cells have no duplicate-free 32-word windows, the sensitivity identifies robustness on the **common retained support**, not the effect of purely deleting duplicate text while every original source remains available. The same-support unfiltered control addresses this selection explicitly.

**Further observed trade-off:** On filtered isiXhosa-view data, PP-WW is -0.200 in 1888 and -0.409 later. Language-view effects differ in sign; making English-view more compact can correspond to making isiXhosa-view less compact. Magnitudes represent *subword counts*, not proven disadvantage to speakers.

**Limitations / reasons not to overclaim:**

1. The reference years are reused from preceding phases at analyst level, so this is prospective *only relative to F6's new scores*, not pristine external confirmation.
2. Duplicate detection is exact 13-word contiguous matching. Paraphrases, approximate OCR duplicates, shorter advertisements and semantic overlap may remain.
3. Entire training page/view cells disappear unequally across views (English disproportionately), even though both conditions share the remaining cells.
4. The held-out archive still shares some exact 13-grams with the complete pre-1888 corpus; duplicate exclusion in training samples does not guarantee source-independent evaluation.
5. No inference about actual article genres, sentiment, political worldviews, modern isiXhosa, LLM applications or social fairness follows from these experiments.
6. Issue bootstrap measures conditional archive variation; five tokenizer seeds are technical replications, not independent newspaper populations.

**Decision:** Accept F6 for the frozen computational contrast. Do not retune or widen the claim. Preserve the null and heterogeneous comparisons (especially within-view and xh seed signs). Move to Phase 5 only for a precommitted downstream test of bits per character (BPC) using frozen models and global validation-selected smoothing alpha=100.

**Research trace:** `Phase4/PRECOMMIT.md` → `Phase4/outputs/TRAINING_LOCK.json` → `heldout_scores.csv` → `f6_contrasts.csv` → `PHASE4_DECISION.json` → `PHASE4_INDEPENDENT_VERIFICATION.json` → `PHASE4_FRESH_RETRAIN_CERTIFICATE.json`.
