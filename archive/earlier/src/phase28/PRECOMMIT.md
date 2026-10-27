# Phase 28 - topic-proxy controlled newspaper comparison

Frozen 2026-10-24, before computing any Phase 28 outcome. This is a **diagnostic sensitivity study**, not an independently held-out confirmation: Phase 26 already evaluated 1888, and the archive's human-coded 1885 article-type/topic labels are unavailable locally. The official archive website shows a browser verification challenge and cannot be ingested in this environment. Do NOT call the proxy 'human genre labels', 'political viewpoint' or 'cultural ground truth'.

## Scientific question
Do the newspaper-domain signatures (and BPE source-specialization) reported in Phase 26 persist when compared excerpts are aligned by publication month and an unsupervised lexical content proxy, instead of selecting pages without reference to content?

## Frozen inputs
- Phase 26 `windows.jsonl` (581 fixed 90-word body snippets, exact duplicate snippets removed), `newspaper_test_predictions.csv` (char_wb and word class predictions from 1885+1887 training on held-out 1888), and `tokenizer_page_scores.csv` (both source-trained BPE models, 3 seeds, on 1888).
- The saved predictions and BPE costs are reused verbatim; no classifier/tokenizer retraining, no change to 1888 scores.
- Label metadata are year, month, newspaper, issue. The 1885 labels described in Schoots et al. (2023) are not in these files. Under no circumstances use document source/year as a replacement for genre.

## Match construction (no outcome used to select pairs)
- For **each year** and **each calendar month**, exactly pair Imvo and Isigidimi body excerpts 1:1 without replacement using global Hungarian minimum-cost matching.
- Similarity: word-unigram TF-IDF on source-cue-masked historical surface text (mask full tokens containing `imvo`, `isigidi`, `samaxosa`, `zabantsundu`, `bantsundu`, `ntsundu`, `mnyama`, `xosa`, `jabavu`, `gqoba`, `makiwane`, plus `lovedale`, and `mr`); remove the highest corpus-frequency common forms using unsupervised `max_df=0.35` and `min_df=2`, `sublinear_tf=True`, `norm='l2'`, `max_features=15000`.
- Fit the TF-IDF dictionary/IDF **only on 1885+1887** examples, without source labels; transform 1888; do NOT tune matching based on source-classification predictions or BPE scores. Similarity cosine.
- Control: 500 random, month-stratified 1:1 pairings with the same total pairs and equal newspaper/word count, seed=28024; include a no-match full held-out corpus baseline.
- Save every matched pair and similarities, and distinct-issue statistics. No semantic relevance/genre claims from a nonzero similarity.

## Fixed tests and decision rules
H28a (matching feasibility): average pairwise cosine similarity exceeds random-pair mean by >=0.025 AND the month-cluster bootstrap 95% CI for paired difference is strictly above 0. Otherwise matching did not supply a meaningful lexical-content-control proxy and inference from the next outcomes is especially limited.
H28b (source signature remains detectable): Phase 26's frozen char_wb source classifier, when scored on 1888 **lexically matched** pages, has an issue-balanced accuracy above 0.65 and its month-cluster bootstrap 95% CI lower endpoint > 0.50. For matched score each issue's available matched pages have equal weight within newspaper. This supports only remaining *publication-source* predictability conditional on the proxy, not a genre-neutral effect.
H28c (tokenizer specialization remains): For 1888 matched pages, compute BPE difference between specialists by source (Imvo: `ISIG_E - IMVO_E`; Isigidimi: `IMVO_E - ISIG_E`), mean across 3 fixed seeds, score macro issue weighting. For both source groups the bootstrap interval over 12 months must have lower endpoint >0. If either fails, do not claim specialization survives matching.

Report within-year (1885, 1887, 1888) matching feasibility, primary accuracy on 1888, page/macro-issue metrics, the random-pair calibration distribution, bootstrap 2000 replicates (seed=28025), short-form diagnostics for source metadata leakage, and a reproducibility verifier. Negative results and limitations are mandatory. Match quality must be reported whether hypotheses pass or fail. No paper drafting.

## Inferential limitation
The external scholar-coded article types and topics are unavailable and cannot be aligned. Topic proxies may track style, orthography or genre imperfectly. The 1888 test is reused from Phase 26 and cannot be claimed independent. No causal effect of newspaper ownership, cultural change or human ideology can be concluded. Source years 1885, 1887 and 1888 are the only overlap; missing 1886 Isigidimi cannot be imputed.
