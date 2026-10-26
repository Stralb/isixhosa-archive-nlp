# Phase 21 prospective protocol — character-tolerant candidate retrieval

**Frozen on 2026-10-24 before generating or evaluating Phase 21 queries and before looking at any Phase 21 retrieval scores.** Prior Phase 18/19/20 protocols and outcomes remain untouched. This is not human-intent retrieval or an authentic OCR benchmark.

## Scientific uncertainty

Phase 20 found a noisy five-word excerpt retrieval candidate-recall bottleneck for word BM25, but did not evaluate a computationally completed character-tolerant full-corpus retriever. Question: can a strong, *conventional*, tokenizer-tolerant character-trigram retriever recover known evidence under controlled synthetic corruption, with an equal top-20 retrieval budget, and does that improve final full-page fuzzy Top-1?

## Independent issue-origin partition; exact evidence

- Searchable index: all 3,608 supplied raw historical newspaper pages. Any source-view field is an imperfect archive designation, not a linguist-verified label.
- Exclude every query-origin issue previously used in the Phase18 balanced benchmark, Phase19 and Phase20. Their documents remain searchable as ordinary archive pages. No previous test results used for method selection beyond motivating the question.
- Sample **30 Imvo + 30 Isigidimi previously unqueried distinct issues**, all using the supplied `xh` source view, after verifying 65–350-word eligible paragraphs, on 2026102421 seeded SHA256 ordering; one eligible paragraph/page per issue. Prefer non-front page. Do not use retrieval scores in selection.
- Fixed per-source sample sizes were selected after a *source-only eligibility check* counted 90 Imvo and 37 Isigidimi eligible unused issues, without building or scoring queries.
- Paragraph must appear in an original raw-page paragraph; select a 13-word within-paragraph exact evidence span. Gold: source page plus every archive page sharing that 13-word within-paragraph evidence sequence (precomputed before rankings). Do not treat topical nonmatches as disproven relevance.
- Create five paired forms from one source paragraph: `keyword` = six nonconsecutive distinctive surface words spaced across the paragraph (no copied 3-word query sequence); `clean` = 5 adjacent center words from the 13-word evidence span; `edit1` = clean query with 1 interior alphabetic character substitution; `edit2_boundary` = 2 interior substitutions on distinct tokens and 1 split-or-join word boundary operation; `edit3_boundary` = 3 interior substitutions on distinct tokens and 1 split-or-join. All deterministic using SHA256 salted seeds; no corrected isiXhosa spelling, no invented language annotations.
- 60 issues × 5 query forms = **300 queries**, same gold per issue. If eligibility unexpectedly fails, record amendment before rankings, never replace a result-driven sample.

## Fixed lexical/retrieval policies

Identical lowercased alphabetic surface-form lexical analysis (without orthographic normalisation/stemming) and full 3,608-page BM25 index (k1=1.2, b=.75, Phase20 implementation), with original BM25 score tie-breaks and deterministic page index ordering.

Build a **full-corpus within-word padded character-trigram TF-IDF index** without `max_features`/query-dependent vocabulary tuning: for each indexed lowercase alphabetic word, count trigrams from `'^'+word+'$'`, aggregate word counts into each page; apply TF-IDF (smooth IDF, sublinear TF, L2 row normalization) and cosine query/page similarity. Query trigrams use the same procedure without OOV word filtering. No test-derived fitting of hyperparameters; fitting the IDF on the searchable *unlabeled archive* is standard index construction. This is deliberately a standard strong lexical baseline, not a novel model.

Compare equal candidate budgets of 20 pages:

1. `word20`: raw BM25 top20, BM25 rank order.
2. `word20_fuzzy`: same raw20, Phase20 normalized-no-space RapidFuzz `partial_ratio` against complete raw page, ties by raw BM25.
3. `char20`: character-trigram TF-IDF top20, cosine rank order.
4. `char20_fuzzy`: same char20 reranked by identical full-page fuzzy.
5. `hybrid15w5c`: 15 raw BM25 + 5 character candidates, unique with raw/char backfill to exactly 20, rank by reciprocal-rank fusion (k=60) over both full ranked lists.
6. `hybrid15w5c_fuzzy`: exact `hybrid15w5c` pool, full-page fuzzy, ties by raw BM25.
7. `xh20_fuzzy`: silver source-view xh BM25 top20 and full-page fuzzy, as a legacy hard-language-field control. This is not authoritative language-ID routing.

All retrieval/candidate policies include the target page in the document index. No candidate budget exceeds 20. Character preprocessing intentionally ignores cross-word ngrams; test as designed, not as a general claim about all char n-gram methods. Always report both **candidate evidence recall@20** and evidence Top-1/Top-5; originating-page Top-1 is secondary.

## Prospective hypotheses/gates

- **H21a — candidate rescue:** `char20` minus `word20` on `edit2_boundary` known-evidence recall@20 is **at least +10 percentage points**, and the paired source-issue bootstrap 95% lower bound is **strictly above zero**.
- **H21b — retrieval impact:** `hybrid15w5c_fuzzy` minus `word20_fuzzy` on `edit2_boundary` evidence Top-1 is **at least +5 percentage points**, with bootstrap 95% lower bound strictly above zero, **and** neither newspaper group declines by >5 percentage points. If only char20_fuzzy succeeds but hybrid does not, H21b FAILS; no retrospective swapping.
- **H21c — keyword nonharm:** on `keyword`, `hybrid15w5c` (fusion ranking, no fuzzy) must be no more than **5 percentage points below** `word20` evidence Top-1 within *either* newspaper. This is a safety gate for a general candidate+ranker, not proof of good natural keyword relevance.
- Primary reports: newspaper-equal macro for each endpoint. Paired issue-stratified bootstrap, 10,000 iterations, seed 2026102422. Report subgroup outcomes, uncertainty, standard ranking controls, all five query forms, and failures. Only H21a/H21b/H21c prospectively tested; other comparisons are exploratory. No significance interpretation beyond source-issue/sample construction.
- Null control: permute gold evidence anchor association between sampled issues **within newspaper**, fixed seed 2026102423; compare Top-1 frequency to ensure nondegenerate source-prior results.

## Integrity and interpretive boundaries

- Verify each source issue excluded from all prior query origins and used exactly once. Five query forms share fixed gold; each 13-token span appears fully within a **single** raw paragraph; do not silently join boundaries; keyword does not reproduce contiguous 3 words.
- Verify clean and each edited query differ by the prescribed perturbation counts, anchor included in multi-gold, index and score arrays cover exactly 3,608 pages, candidate sizes capped at 20, no test-condition tuning or refitting after outcomes.
- Gold is phrase-matching known evidence, not independent relevance. Query fragments are source-derived; errors are synthetic, not observed historical OCR. Do not call this a human/OCR benchmark, a causal OCR demonstration, a fairness estimate, or a novel NLP retrieval method.
- If no reliable authentic OCR alignment and independent human-authored needs exist, state those as outstanding, not as implicit assumptions. Even positive gains cannot support publication-method novelty without comparison to stronger modern baselines and independent evaluation.

## Research decision

If H21a passes but H21b fails, characterize an *index coverage* improvement without Top-1 benefit. If both pass with H21c, retain as strong conventional baseline worth authentic external validation. If they fail, reject char trigram rescue in the specified protocol. No new report/abstract unless requested.
