# Excellence Sprint Phase 5 — downstream validity: execution lock

Date: 2026-10-24, before computing Phase 5 held-out language-model losses.

## Authoritative frozen decisions

Read `Phase1/FROZEN_PROTOCOL.md` F5 and `Phase4/PHASE5_START_PROMPT.md` first. Neither is changed. Source ZIP SHA256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`, Phase 1 protocol SHA256 `bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4`. Phase 2 selection fixes `alpha=100`, selected before 1888+ outcomes. No other smoothing variants scored or chosen based on test. Forty prefit SentencePiece models from Phase 2 must match `model_training_manifest.csv` SHA256 and original vocabulary sizes. No model training/retraining, no reuse of Phase 4 models in primary F5.

## Inputs and estimand

Recompute Phase 2 `build_lm_docs`: all qualifying Imvo 1885 / 1887 Jan-Sep page-view text, first 128 `[a-z]+` tokens of each view with >=40 tokens. Check SHA256 of *every* LM document against the original Phase 2 `lm_training_manifest.csv`. Same documents in same order for **every tokenizer**. Train Phase 2 count `SmoothedBigram` on each tokenizer's segmentation of those fixed raw strings; reset previous token to -1 at each document boundary. Calculate on every Phase 3 page-view text selected by `load_heldout` (identical hash, chars, view, issue, year and period manifest): negative sum log2 of predictive probabilities, divided by the same number of alphabetic characters. No extra BOS/EOS token, no unigram interpolation change, no model-specific document selections. Do not fit a neural model or infer neural quality.

Fixed `SmoothedBigram` formula: unigram `base=(count(t)+0.1)/(N+0.1*|V|)`; predictive probability `(count(prev,t)+100*base)/(count(prev)+100)` with `prev=-1` at start of each document. Cost is `-sum log2(prob)` for each token. Count vocab is full SentencePiece vocab including unused piece IDs. Language view = archive's EnTxt/XhTxt, not verified labels.

Primary F5 BPE English 1888 factorial allocation: `Delta_A=0.5*((BPC_WP-BPC_PP)+(BPC_WW-BPC_PW))`; **SUPPORT** only if effect <=-0.010 bits per alphabetic character, 95% paired issue bootstrap upper bound strictly <0, and 1889-94 mean effect <0. Otherwise **NOT SUPPORTED** even if one confidence interval excludes zero or the piece-count outcome F1 succeeded. Report F5 as one directional precommitted test only, not a new significance family. Secondary same-factor BPC contrasts for all family x period x view; preserve negatives and seed heterogeneity. Compute `within_view`, `interaction`, `WW_minus_PP` via the Phase 3 formulas. On every period use 5000 paired whole-issue bootstrap draws (same issue multiplicities for all 40 tokenizers), fixed RNG `343401` (1888) and `343402` (later), retaining five model seeds as technical replications. Six-year-cluster bootstrap sensitivity later-year, RNG `343501`, 5000 draws.

Training/test exact overlap: join Phase 4's fixed `train_test_overlap_diagnostic.csv` to Phase 3 page-view manifest and report 13-word overlap rates by view/period; also show a **secondary sensitivity** of BPC results on page-view excerpts with zero 13-word matches to the complete *train-period archive* as a conservative diagnostic. Because this subset is selectively filtered, it cannot override F5. The main F5 result remains full 2493 page/view excerpts and is NOT replaced by this subset.

## Integrity gates

- no test-derived hyperparameters or changing hypotheses after results;
- original LM-train manifest SHA per raw string, original Phase 3 test manifest SHA per raw string, 40 model/vocab SHA checks;
- 2,493 held-out page/view excerpts x 40 models, no mismatched evaluation strings or label changes;
- `alpha=100` used for every single score, save per-excerpt raw bits and char denominator, zero unexpected unknowns;
- exact `SmoothedBigram` formula cross-checked against a separately implemented LM, including independently derived first/uni/bigram counters and per-doc start resets;
- check equality of every test score to recomputation within numerical tolerance, and bootstrap quantities independently checked;
- package original frozen materials and all outputs with hashes, do clean rerun/replay from ZIP extraction;
- provide explicit F5 decision and note reused 1888-94 corpus is not independent research-program-level validation.

Expected failure interpretation: F5 may fail despite all four Phase 3 representation hypotheses succeeding. A failure is scientifically meaningful and must not be obscured by exploratory subgroup wins. No paper drafted.
