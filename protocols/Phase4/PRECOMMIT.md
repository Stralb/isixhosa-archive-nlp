# Excellence Sprint Phase 4: repeated-text falsification (2026-10-24)

Status: FROZEN before Phase-4 model fitting or held-out scoring. Derived from Phase 1 frozen hypothesis F6. The already examined 1888-1894 newspaper period is a conditional replication setting, not an untouched external test.

## Question and falsifiable outcome
Is the historical bilingual tokenizer sampling effect mainly an artefact of training on repeated newspaper text? Exclude any candidate 32-word training window containing an exact contiguous 13-word string found in at least two distinct training publication issues; learn new tokenizers without changing the downstream evaluation set. Compare with new unfiltered models sampled over exactly the same eligible page/view support.

## Inputs and intervention
- Archive: `TxtLangFiles1870-1894_TB.zip` sha256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`.
- Baseline science: Phase 1 protocol sha256 `bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4`; Phase 2 selected LM alpha=100 (irrelevant to F6); Phase 3 PP/WW BPE already evaluated.
- Training text ONLY: IMVO 1885 and Jan-Sep 1887, `EnTxt`/`XhTxt` views lowercase `[a-z]+`, >=16 eligible words. Test untouched until all Phase-4 models are fitted and frozen.
- Duplicate detector: exact 13-word shingle with owners in at least two distinct training issue dates, across both archive views. A 32-word start is clean if NONE of its 13-word shingles is in that set. A page/view without clean starts is excluded from BOTH conditions.
- **Condition filtered**: eligible clean starts ONLY. **Condition same-support control**: ALL eligible starts of the very same surviving page/views. Both re-normalize PP and WW sampler probabilities on identical page/view support, using original word counts for sampler weights. This tests window exclusion, not changing which issue/page/views are available.
- Sampler: existing frozen `factorial_sampler.draw()` with 32-word within-page draws, 500,000 +/-30 alphabetic characters for every fit; PP and WW only. 5 seeds 441-445, independent deterministic arm-specific RNG streams.
- Model: SentencePiece BPE, fixed vocab=2000, normalization identity, character coverage 1.0, `num_threads=1`, code `Phase2/scripts/run_phase2.py::train_sp`. NO hyperparameter selection or model changes based on Phase 4 test outcomes.
- Outputs: 20 newly trained models (2 conditions x 2 arms x 5 seeds), training sample hashes, model SHA256, source-support audit, plus training source-line manifests without redistributing original newspaper text.

## Frozen evaluation and inference
- Same Phase 3 held-out source text, first 128 alphabetic words of each `EnTxt`/`XhTxt` view with >=40 words on IMVO 1888, 1889-1894. Assert identical filename/view,text-SHA256 against Phase3 `heldout_pageview_manifest.csv`.
- Primary response: `E=100 * pieces / alphabetic characters`; zero unknown tokens required as a sanity check. Aggregate corpus-character weighted within view, paired between conditions and arms.
- Effect `R=E_PP-E_WW`: positive means PP uses more subwords, i.e. word-proportional training yields more compact representation. Define `R_filtered` and `R_control` on shared support; show `R_original_Phase3` separately as non-randomized prior comparator. Also report `R_filtered - R_control` diagnostically, and both archive language views.
- F6 preregistered acceptance: for **1888 English view**, `R_filtered>+0.30 pieces / 100 chars`, the entire paired 5000-draw whole-issue bootstrap 95% CI >0, and 1889-94 English view `R_filtered>0`. Otherwise F6 rejected/inconclusive as appropriate. Sign of 5 seed effects required in report, but not an additional acceptance gate. The same-support control is a confound-control contrast, not the denominator of the F6 decision.
- For two periods use paired bootstrap of whole publication issues (draws=5000, seeds=343401 for 1888, 343402 for later); keep 5 model seeds fixed, report issue-cluster uncertainty and six-year aggregate sensitivity as auxiliary. All selected comparisons, successful and unsuccessful, are reported, without post-outcome tuning.
- After primary scoring, perform independently coded source-text extraction verification, train/test shingle-overlap sensitivity (not used in model selection), plus independent recomputation of all 20 x 2493 scores and bootstrap CIs. Additionally verify the clean-training samples actually contain no banned shingles, and control samples are from the identical eligible page/view support.

## Interpretation and stop rule
This is a **conditional, preregistered falsification** developed after the historical outcome years had already been examined in earlier research. It may strengthen robustness evidence but cannot prove zero contamination, cultural bias, sentiment, social harm, improved semantics or independent historical transfer. If filtered models fail F6, preserve negative result and do not retune. Phase 4 does NOT include Phase 5 language-model work or paper drafting.
