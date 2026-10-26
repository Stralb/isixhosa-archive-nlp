# Excellence Sprint Phase 4 gate — PASS

Date: 24 October 2026. Scope: only F6 from original Phase 1 preregistration. Phases 1-3 are untouched.

**Gate status: PASS. F6 SUPPORTED in the preregistered, conditional historical setting.**

## Primary F6 decision

`R = E_PP - E_WW`, with `E = 100 * SentencePiece pieces / alphabetic characters`. Positive R favors word-proportional sampling (WW). The **filtered** tokenizer pair was trained with every 32-word window containing a 13-word sequence repeated across training publication issues excluded.

| English-view evaluation | Filtered R | 95% paired publication-issue bootstrap CI | Seed sign |
|---|---:|---:|---|
| Imvo 1888, 52 issues | +2.389 | [+2.182, +2.590] | 5/5 positive |
| Imvo 1889-1894, 306 issues | +2.119 | [+2.050, +2.185] | 5/5 positive |

**Precommitted F6 success:** 1888 R >+0.30 and bootstrap lower bound >0, later-year R>0. All criteria met. This does not itself establish a meaningful downstream loss or social harm.

## Valid controls and negative results

- The original archive contained 595 eligible training page/views. Exact training-only 13-word reprint exclusion retained 533: 198/254 English and 335/341 isiXhosa. **56 English and 6 isiXhosa page/view cells were lost**. Both filtered and unfiltered controls use the same surviving support so they are directly comparable, but the result is conditional on that selected support.
- Same-support unfiltered English PP-WW R = +2.097 [+1.905,+2.280] in 1888; +1.615 [+1.540,+1.687] in 1889-94. Thus the sampling effect does not require training-line exclusion and is not merely created by the selection of surviving support.
- Filtering changes the English PP-WW gap relative to its same-support control by +0.292 [+0.164,+0.421] in 1888, +0.504 [+0.465,+0.542] later. This *follow-up comparison* was not a stand-alone preregistered primary hypothesis. Four of five seeds show the positive change; do not imply seed unanimity.
- isiXhosa PP-WW R in the filtered condition is -0.200 [-0.380,-0.030] in 1888, -0.409 [-0.469,-0.350] later. This reverses the English sign, consistent with a representational trade-off, although only 3/5 seeds have the expected negative sign on 1888 isiXhosa. Do not claim seed-stable parity or fairness gains.
- Repeated training-only 13-gram types identified: 16,691; removed from all filtered sample lines: **0 observed violations**. Unfiltered matched controls contain 98,181 occurrences of the banned 13-grams across sampled 32-word training lines (overlapping counts; not distinct articles).
- Full *training-source to evaluation* 13-gram overlap diagnostic is not zero: 1888 English 3.60%, 1888 isiXhosa 4.19%, 1889-94 English 2.92%, isiXhosa 1.12%. Therefore duplicate-window exclusion does NOT prove the held-out data have no verbatim overlap with the entire archival training pool.

## Verification outcome

PASS: 20 new BPE models; 55,850 training lines independently redrawn with identical SHA256; source support independently reconstructed; 49,860 held-out tokenization counts independently re-encoded; 20 bootstrap contrasts independently recomputed (largest numeric difference 4.44e-16); 2,493 training/evaluation 13-gram overlap rows independently recomputed; zero unknown tokens. Clean retraining of all 20 models from source reproduced all 20 learned vocabularies, all 49,860 scores and all 20 contrasts exactly. Model binaries can contain training path metadata; path-independent learned vocabularies and output parity are the strong replay invariants.

## Scientific boundary

This is a precommitted follow-up after other phases had already analyzed the same Imvo 1888-94 years; not an untouched external replication. `EnTxt`/`XhTxt` are archive-provided content block views, not human linguistic truth, and only Imvo is in this factorial study. No article genre, political identity, sentiment, or social harm has been established. Phase 5 downstream test is not included; the user has not requested a paper.

**Phase-4 completion condition met. Next: Phase 5, frozen alpha=100 bigram modelling on identical raw LM text, with all negative outcomes retained.**
