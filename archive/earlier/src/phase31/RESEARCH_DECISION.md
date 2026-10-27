# Phase 31 scientific decision — Out-of-period BPE and context coverage
24 October 2026 | Evaluated after frozen precommit; no paper drafted

## Question
Do newspaper-specific BPE segmentation differences persist **outside** the 1885/1887 training years, and do they produce a meaningful reduction in how much historical material fits a model's finite token budget?

## Data and controls
- Isigidimi `XhTxt` **1870–1884**: **1,088 eligible pages / 164 issues** from 1,255 available pages; 167 excluded for lacking a 90-word single paragraph.
- Imvo `XhTxt` **1889–1894**: **774 eligible pages / 305 issues** from 1,225 available pages; 451 excluded by the identical rule.
- All windows exactly 90 lowercased ASCII-letter surface forms from the first 90 words of the longest paragraph; newspaper source label is from filenames, not linguistic ground truth. This favors pages containing a sufficiently long paragraph and may over/underrepresent advertisements or genres differently in each newspaper.
- Reused Phase 30 frozen BPE-1000 source models (3 paired seeds), plus A/B balanced-source placebo models trained on intact equal-character-source passage allocations. No retraining, retuning, or 1888 test-set re-evaluation to select settings.
- Newsprint histories **do not overlap in these two cohorts**. We estimate within-source contrasts only. The older Isigidimi corpus is not a time-matched control for later Imvo.

## Findings
**H31a supported.** On **out-of-tokenizer-training-period** issues the specialist model's savings relative to the opposite newspaper's specialist is:
- Early Isigidimi: **+0.2562 pieces / 100 alphabetic characters**, issue-bootstrap 95% CI **[+0.1830, +0.3271]**.
- Later Imvo: **+0.4980 pieces / 100 alphabetic characters**, issue-bootstrap 95% CI **[+0.4126, +0.5850]**.
This is evidence of transfer of a *small* tokenizer specialization effect to earlier/later issues within the same archive, not an independent external-corpus replication.

**H31b NOT SUPPORTED.** At a preregistered 256-piece context limit, domain-specialist models retained on average just **+0.0194 words** (CI [+0.0025,+0.0386]) out of 90 on Isigidimi and **+0.0035** (CI [-0.0313,+0.0401]) on Imvo versus cross-source models. Neither approached the chosen **+2 words** practical-effect threshold. Almost all 90-word windows fit already; this ceiling makes 256 an insensitive stress test for these lengths.

**H31c NOT SUPPORTED.** Balanced-source controls did not improve the worst-source 256-token retained-word regret over the **better** of two fixed single-newspaper tokenizers. Best fixed-model max regret was **0.0035 words**, balanced was **0.0192 words**. Balanced-minus-better-fixed disadvantage is tiny and should NOT be interpreted as harm. A 50% relative improvement is not meaningful when the comparator shortfall is nearly zero.

**Precommitted budget sensitivity:** At **192 tokens**, specialist-minus-cross-source retained words was **+0.430** (Isigidimi) and **+0.969** (Imvo); at **224 tokens**, +0.292 and +0.425; by 288–320 tokens effects approach zero. These are secondary chosen-budget results (not independent confirmation); all remain under the two-word precommitted practical threshold when interpreted against the primary decision.

## Scientific contribution and boundaries
1. Earlier work established *cost* asymmetry and model-probability downstream shifts. Phase31 tests a distinct **operational implication**: lost readable content under fixed token budgets. That implication **does not hold at practically important magnitude within the frozen 90-word/256-token test**. This is a valuable *falsification* of an intuitive but too-strong Responsible NLP argument.
2. The small absolute token-cost effect survives independent publication years but weakens compared with earlier 1888 held-out excerpt findings; comparisons differ in dates, paragraph sampling and source mix, so do not attribute the change to historical evolution.
3. This work does not show historical sentiment, cultural evolution, domain fairness, or exclusion in deployed transformer retrieval/generation systems. Tokens from a tiny 1,000-piece BPE are not modern LLM tokenizers, and the 90-word/256-piece operating point was an experimental choice rather than a production model limit.
4. No human genre/topic annotations are available in the supplied source. Two publicly described books are different genres/editions and are not reliable chronological comparators without exact digital-transcription provenance. No external book results were fabricated.
5. The experiment is not an independent external dataset: both evaluation cohorts are from the same digitized IsiXIT collection as training, although their years and individual pages are disjoint from Phase30's tokenizer training.
6. Overall, **claims that tokenization causes cultural misinterpretation or practical historical text exclusion are not supported by these results**. Retain the narrower and reproducible source-composition -> frequency -> segmentation mechanism as the mature contribution. Revisit practical harm only when a relevant downstream task/context budget provides a justified operational threshold.

## Next decision
**Stop artificial extensions of the same tokenizer observation.** One genuinely new contribution requires a distinct dataset/genre with licensed and version-matched historical text, a validated downstream task, or the existing 1885 article-level type/topic labels. Avoid another arbitrary n-gram or budget search to obtain a success threshold. Publication writing can now frame a careful methodological case study rather than claim unobserved social consequences.

## Reproducibility
- `PRECOMMIT.md` and `ESTIMAND_CLARIFICATION_PRE_EVAL.md` were saved before scoring.
- `FROZEN_INPUT_SHA256.txt` and `RESULTS.json` contain input/model/cohort/ranking hashes.
- `INDEPENDENT_VERIFICATION.json`: 432 scores recomputed from 36 independently extracted archived pages; full per-issue summary independently recomputed. Zero model/source hash discrepancies; max numerical difference ~7e-15.
- `page_model_scores.csv` retains per-model per-seed costs for **22,344 model-window evaluations** (1,862 pages × 3 seeds × 4 frozen tokenizers).
- Text of primary newspaper archives is not republished in this package; rerunning requires the original `TxtLangFiles1870-1894_TB.zip` attached by the user.
