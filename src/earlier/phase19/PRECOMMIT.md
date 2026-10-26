# Phase 19: precommitted nonverbatim English evidence-search transfer pilot

Timestamp: 2026-10-24, before any retrieval ranks on new queries are computed or inspected.

## Scientific question
Does Phase 18's **fixed** language-gated candidate augmentation outperform strong unrestricted BM25 and BM25+fuzzy retrieval on new, nonverbatim factual/keyword archive searches, rather than verbatim five-word source excerpts?

## Fixed sampling and intent boundary
- Archive is 3,608 original raw pages, with exact paired `en` / `xh` source views provided by the archive. The indexed archive includes all pages, including source pages for the queries; this is the retrieval collection, **not train/test contamination**.
- A total of 55 new **assistant-composed** short English keyword searches (40 from Imvo, 15 from Isigidimi) are fixed in `queries_precommit.jsonl`. Each was drafted after *reading an English-view paragraph*, but **without looking at retrieval ranks or outcomes**. It is **not** a corpus of independent human information needs and is **not** multilingual semantic search or genuine isiXhosa evaluation.
- Every source issue is disjoint from **all** sampled Phase 18 English query issues; a query's source issue is unique within Phase 19. 1890-1894 for Imvo, 1871-1888 for Isigidimi. All selected source pages are nonfront. Isigidimi has limited English-view material, so counts are uneven; report each source and equal-source macro separately.
- Exclude an assistant query if an exact three-word sequence from it occurs in its source paragraph, before freezing. Frozen query text and paragraph audit satisfy this check. No new selection/tuning after test access.
- Some sampled Isigidimi material is an announcement or advertisement; do not call the dataset all editorial content.

## Gold, what it does and does not prove
- Anchor gold: the source page used to compose the query, fixed before retrieval.
- Passage-equivalent multi-gold: any **raw page** containing a 13-word contiguous **within-paragraph** sequence fixed from the center of that source's English-view paragraph, tokenized with Unicode-letter and apostrophe pattern as in Phase 18. Source anchor must be one of the gold pages; assert this.
- This avoids falsely penalizing verbatim reprints of the known passage. But pages that answer the question in different words may remain unlabelled: **this tests known-evidence location, not exhaustive semantic relevance or answer accuracy**. Preserve original-page results as a stricter descriptive check; do not call off-anchor retrieval necessarily irrelevant.

## Fixed retrieval policies, fair budget
- BM25: same Phase 18 CountVectorizer word-index, full-vocabulary k1=1.2, b=.75, full 3,608-page document index, **no tuning**. Query term presence binary, document TF respected. Each method retrieves at most 20 pages.
- `raw_bm25`: unfiltered raw top20, score-order no fuzzy (essential competitive lexical comparator).
- `raw_fuzzy`: unfiltered raw top20, same RapidFuzz `partial_ratio` on ASCII alnum normalized query/page, with raw BM25 and document-ID tie breaking (Phase 18 strong comparator).
- `hard`: use the **unchanged Phase 18 short-query LID design** to select *only* Xh or En field, top20 fuzzy.
- `gated`: if short-query classifier predicts English, union 15 raw + 5 En field candidates (raw backfill to 20); otherwise raw top20, fuzzy. No threshold, no issue-based tuning.
- `unconditional`: always union 15 raw+5 En, fuzzy.
- `oracle_english_field`: known English source view, En-only top20, fuzzy; diagnostic **not deployable**.
- `rrf_raw_en`: fixed k=60 reciprocal-rank fusion of unfiltered raw + En field ranked full page lists; select top20; identical fuzzy rerank. Existing conventional fusion baseline; no tuning.
- `char_word_tfidf`: character word-boundary 3-5 gram TF-IDF on raw archive, top20, identical fuzzy rerank; score if resource feasible and document omission otherwise.
- Candidate budget is always at most 20, no change per condition, all sources use same reranker.

## Short language-ID model
- Refit the Phase 18 char 3-5 gram, balanced Logistic Regression C=2 on two deterministic random 5-token excerpts per eligible Imvo `en`/`xh` language view in 1884-1887, excluding the 408-query Phase18 **held-out issue list** and all Phase 19 source issues. Source view is a silver field label, not human language ground truth.
- The intent and word length distribution differs from its five-word training; this OOD shift is part of the test, not a tuning opportunity.

## Primary metric, uncertainty, and decision
- Primary: equal-source macro **Top-1 any passage-equivalent gold page**, with two-source issue-grouped paired bootstrap (10,000 resamples, seed 20261024); also report source-specific hits and source-issue confidence intervals, original-anchor hits, Top-5, candidate recall@20.
- Prospective H19a: `gated` beats **both** `raw_fuzzy` and `raw_bm25` by at least +3 percentage points and lower bootstrap limit >0 on the fixed nonverbatim search sample, with no source-specific reduction >2 points. Do not accept source-general or novel method if any check fails. RRF/char baseline may refute value even if H19a passes.
- Null/control: within-source randomly shuffled gold assignment (issue-disjoint) tests chance-level scoring under unrelated targets; use the same fixed outcomes and a frozen shuffle seed 20261024.
- This is a **55-query source-anchored assistant-authored pilot** without semantic gold adjudication or isiXhosa expert; even a positive result would warrant independent human relevance work and a strong pretrained multilingual dense/hybrid baseline before publication claim.
- Report adverse/negative results as-is. No paper writing.
