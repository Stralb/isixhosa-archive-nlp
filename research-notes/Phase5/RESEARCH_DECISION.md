# Excellence Sprint Phase 5 — scientific research decision

**Outcome: F5 SUPPORTED**, conditional on the historical Imvo corpus, the prespecified count-bigram implementation, the previously examined held-out years and Phase 2's validation-selected `alpha=100`. All reported effects are improvements/reductions in **bits per alphabetic character (BPC)**, not language understanding or demonstrated social fairness benefits. Lower BPC is better.

## Primary prespecified F5 test (Factor A: archive language-view allocation)

The Phase 1 hypothesis is about the **marginal effect of word-like versus page-like language allocation**, averaged over the two within-language selection policies:

`Delta_A = 0.5*((BPC_WP - BPC_PP) + (BPC_WW - BPC_PW))`.

For **BPE, English-view**, on 1888 text:

- `Delta_A = -0.037216654819` BPC; paired whole-issue bootstrap 95% interval **[-0.042364443062, -0.032118837492]**; 52 publication issues; 134 English-view page excerpts; five of five tokenizer seeds negative.
- All three frozen F5 gates PASS: magnitude at least **0.010** BPC, interval upper bound below zero, and later-year same negative sign.
- 1889–1894: `Delta_A = -0.025192380380` BPC; issue-bootstrap interval **[-0.026652223053, -0.023654483173]**; 306 issues; 927 English-view excerpts; five of five seeds negative. Additional six-year-cluster interval `[-0.029507, -0.021212]`, based on just six clusters.
- The same raw 545 LM-training documents (Imvo 1885 and 1887 Jan-Sep), identical raw 2,493 test excerpts and model capacity/vocabulary size were used for all 40 fitted tokenizers. Only the pretrained tokenizer's learned subwords differ between arms. Alpha **100** was globally selected from `{1,10,100}` using duplicate-aware October–December 1887 validation before held-out evaluation.

**Decision: Support F5 only for the isolated archive-view allocation mechanism, with a fixed smoothed count bigram language model.** The practical effect is modest, and does not demonstrate language understanding, sentiment quality, downstream human value or societal harm.

## Dependent algorithm check and negative / counterexample results

- Unigram English-view `Delta_A`: **-0.03834** BPC [**-0.04366, -0.03318**] in 1888; **-0.02391** in 1889–1894. Five of five seeds in both periods support the negative direction. This is *secondary*, not an independently hypothesized primary F5 test.
- **Within-view selection offsets the gain.** BPE English `Delta_B` is **+0.01938** BPC in 1888 and **+0.02970** later. Accordingly, the unseparated complete `WW - PP` change is **-0.01784** in 1888 but **+0.00450** in 1889–1894 (slightly worse, and the opposite of a universal downstream win). Unigram `WW - PP` is **-0.02920** in 1888 and approximately **-0.00038** later, with the later interval crossing zero. This explains why Phase 33's naive whole-policy downstream effect was not a robust overall benefit despite the factorial allocation effect here.
- **The bilingual predictive trade-off is algorithm-dependent.** In 1888 the BPE isiXhosa-view allocation effect is small and negative (**-0.00275** BPC; three of five seeds negative), rather than the positive effect in its segmentation count. Unigram isiXhosa-view allocation is **+0.01048** BPC with five of five seeds positive, indicating worse prediction under word-like allocation; later Unigram isiXhosa-view allocation **+0.00479**. Do NOT claim segmentation inefficiency entails worse predictive performance in every algorithm.
- All **32** factorial BPC contrasts, both time periods, both archive views, both tokenizer algorithms, five seeds and all null/heterogeneous outcomes remain in `outputs/bpc_factorial_contrasts.csv`, with per-seed and per-year effects retained.

## Train/test repetition sensitivity (secondary, not a new F5 acceptance test)

Phase 4's fixed archival train/test overlap table labels any held-out excerpt with an exact contiguous 13-word phrase also found in the full training-period archive. Of 1888 English-view excerpts, **99/134** lack such an overlap; in 1889–94 **558/927** lack one. Excluding overlap-labelled English excerpts as a conservative, selectively restricted *sensitivity* still yields BPE allocation effects of **-0.02874** [-0.03295, -0.02448] BPC (1888) and **-0.02336** [-0.02529, -0.02135] later. This does not eliminate paraphrase, short-repeat or content-selection confounds, and it cannot replace the full-support primary result. See `outputs/no_training_13gram_overlap_bootstrap.csv`.

## Internal validity and limits

1. **Cause being manipulated:** computational probability of sampling archive-assigned English versus isiXhosa page/view text (Factor A), not historical culture, natural languages, authorship or newspaper events. No verified gold standard exists for original archive EnTxt/XhTxt assignments.
2. **Predictive measure:** intrinsic count-bigram LM loss with fixed smoothing. No neural model and no semantic/cultural task has been validated in this phase. Five tokenizer seeds are technical model replications, not independent language populations.
3. **Potential leakage:** train/validation issue dates disjoint; 1888–1894 texts excluded from fitting. However, some test snippets are repeated in earlier newspaper issues, and the research team already examined the 1888–1894 corpus in earlier phases. This is a protocol-frozen follow-up, **not an untouched external research-program-level confirmation**.
4. **Uncertainty:** paired publication-issue (rather than page) resampling, 5000 draws; future-year six-year clusters report a second sensitivity. Neither sampling procedure turns a single newspaper into independent contemporary genres or publications.
5. **Scope:** Imvo only for this factorial study. The assignment's wider Isigidimi comparison remains a historical background study, and no direct bilingual cultural claim follows from a single newspaper's text-view allocation.

## Verification and reproducibility

- All **40** frozen Phase 2 tokenizers and their vocabulary files matched their saved SHA256 fingerprints; all 40 LM fits used the same 545 original raw training documents, independently re-extracted and checked against Phase 2 document hashes.
- All **2,493** held-out snippets independently reconstructed from the original language newspaper ZIP, with exact agreement on text fingerprints, views, periods and alphabetic-character denominators.
- **99,720** raw score rows reproduced with a *separately implemented* count bigram formula; largest absolute raw-bits difference **6.37e-12**, largest BPC difference **8.88e-15**; zero unknown tokens.
- All **32** factorial point contrasts and paired-bootstrap intervals independently reproduced, largest absolute residual **2.59e-15**.
- Original source, Phase 1 protocol, Phase 2 model-selection lock and Phase 5 execution precommit are SHA256 frozen. A new ZIP-extraction replay reproduced **134 score, bootstrap, audit and figure files byte-for-byte**, including the full independent formula verification. No post-hoc smoothing search or test-driven refitting occurred.

**Research decision: F5 SUPPORTED** for the specified computational loss metric and design. The factorization is essential: **language allocation helps English predictive efficiency in a small count model, while within-view selection can cancel that gain.** This is defensible evidence about archival sampling policy, not measured cultural inequity.

**Future:** Phase 6 is a separate task, covering adversarial review and the mandatory coursework package; **no paper drafted in Phase 5**.
