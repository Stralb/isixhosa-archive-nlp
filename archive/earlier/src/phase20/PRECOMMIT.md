# Phase 20: paired query-form transfer on previously unused newspaper issues

**Frozen 2026-10-24 before constructing Phase 20 queries or inspecting their retrieval scores.** Phase 18 and Phase 19 remain unchanged. This is a new prospective experiment, NOT fresh human semantic-query validation.

## Narrow scientific question

Holding the originating issue, evidence paragraph, retrieval index, candidate budget, and gold evidence constant, does **query form** explain why unrestricted BM25 + whole-page fuzzy matching is effective on corrupted copied passages but fails on sparse nonverbatim keywords? Can a conventional evidence-local lexical reranker improve nonverbatim known-source recovery on new issues?

## Phase 19 development (fully retrospective, not Phase 20 evidence)

A development-only evaluation of local term coverage with windows of 120 word tokens was run on the 55 **already evaluated** Phase 19 queries, comparing fixed window-coverage-first, BM25 × (1+{0.25,0.5,1,2}×coverage), BM25 and full-page fuzzy. Development anchor Top-1 macro: 66.67% coverage-first, 66.25% BM25×(1+2cov), 50.42% raw BM25, 27.92% full-page fuzzy. We SELECT coverage-first with BM25 tie-break **before** the new Phase 20 outcomes. Other variants are not tuned on Phase 20. The development result may overfit Phase 19 and is not independent confirmation.

## Data and deterministic query construction

- Corpus: exactly 3608 supplied newspaper pages, including original raw plus copied `en` and `xh` archive sections. Archive field membership is a *silver source view*, NEVER authoritative language annotation.
- Exclude **all source issues in the 932 Phase 18 balanced queries** and **all 55 Phase 19 source issues** from Phase 20 query origins. Reuse the entire searchable page collection as the fixed index (standard retrieval setting); no train/test *label* reuse.
- Select 160 separate source issues: Imvo English view = 50; Isigidimi English view = 10; Imvo isiXhosa view = 50; Isigidimi isiXhosa view = 50. One page / paragraph anchor per issue, without cross-cell issue overlap.
- Fixed seed = 2026102420 for group selection. For each issue, select an eligible 65-to-350-word copied source-view paragraph from a nonfront page where available, with an intact 13-word span inside a single raw-page paragraph and enough distinct content words for a spaced query. Sampling is deterministic by issue/page/paragraph hash; never optimize using retrieval results.
- Build **three queries per same anchor** (480 queries): (a) `keyword`: a 6-word source-derived keyword list selected from nonadjacent content words, with no exact 3-word query substring in the originating paragraph; (b) `clean`: five consecutive words inside a 13-word center anchor; (c) `synthetic_noise`: same five words, with 2 seeded interior character substitutions on different eligible tokens and one randomly seeded word split/join as in Phase 18F. Three query forms share identical originating issue, page, 13-word evidence span and retrieval gold.
- Generated keyword strings are *automatic source-derived proxies*, NOT human information needs, semantic paraphrases, or independent linguistic annotations. No attempt at translation or correction of isiXhosa forms. The clean/noisy pairs repeat Phase18-style tasks on new issues, not authentic OCR.
- Gold = all pages with an exact 13-word contiguous within-paragraph match to the fixed anchor, including the anchor page; primary outcome is the originating-page Top-1, secondary Top-1 any 13-word-evidence-equivalent page (multi-gold). These are attribution/evidence matching outcomes, NOT exhaustive topical relevance judgments. Gold is defined BEFORE retrieval scores.

## Frozen index/policies and candidate budgets

- Full-page raw BM25 is the identical Phase 18/19 CountVectorizer formula, k1=1.2 b=0.75, 3608 pages, fixed full vocabulary, top20 candidates. No changing candidate limit across methods.
- Comparator A `raw_bm25`: BM25 top20 in original score order (strong lexical comparator).
- Comparator B `raw_fullfuzzy`: same raw candidates reranked with identical Phase18 normalized no-space RapidFuzz `partial_ratio` against full raw page (tie by original raw BM25).
- Comparator C `gated_fullfuzzy`: Phase18 language-ID gate, add 5 English-view candidates to 15 raw when EN predicted, otherwise 20 raw; fuzzy as above.
- Comparator D `unconditional_fullfuzzy`: always 15 raw+5 English-view with raw backfill, fuzzy.
- Comparator E `hard_fullfuzzy`: language-ID-selected single field, top20, fuzzy.
- Comparator F `raw_local120`: exactly raw20 candidate pool, rank by maximum fraction of **distinct query token types of length >=3** appearing within any contiguous 120 raw word-token window, with original raw BM25 score as tie breaker; choice frozen from Phase 19 development. No language expert or learned parameter.
- `gated_local120`: gated candidate pool with the same local reranker, exploratory candidate/enrichment decomposition.
- `raw_windowfuzzy120`: best normalized fuzzy partial ratio across fixed 120-word windows / stride 60 among same raw20, using raw BM25 ties, exploratory passage-fuzzy control. Computational resource may cap this *optional* control; omit rather than replace if infeasible.
- All rankers use same candidate budget; char TF-IDF retriever and multilingual pretrained dense retriever are NOT included unless completed as genuine methods. No model is available locally for a pretrained dense comparator; absence is a limitation, not permission to invent performance.
- Short-query LID: same char3-5 TF-IDF / balanced LogisticRegression C=2 config as Phase 19, trained only using Imvo 1884-1887 development issues excluding all Phase18 balanced query issues, all Phase19 source issues and all Phase20 source issues. Silver view classes only. Report predictions on keyword/clean/noisy by source-view.

## Locked hypotheses, endpoints, uncertainty

- H20a (query-form moderation): mean paired source-view macro [(fullfuzzy−BM25) on `synthetic_noise` − (fullfuzzy−BM25) on `keyword`] >= +10 pp **and** 95% issue bootstrap lower bound > 0. This measures query-form sensitivity, not a new algorithm.
- H20b (transfer of P19-dev ranker): on **keyword**, `raw_local120−raw_bm25` >= +5 pp, 95% source-view/issue bootstrap CI lower bound >0, and no single source-view group loses more than 5 pp. This is a strong **gate**, not guaranteed success.
- Exploratory/diagnostic: benefits of gated versus raw fullfuzzy, clean/noisy results, per-source/view effects, multi-gold and Top-5, candidate recall@20, held-out query quality, and which mistakes are recovered/displaced. Do not declare new method novelty from those.
- Bootstrap resample **source issues**, paired across query forms, with strata for the four source × archive-view cells and B=10000, seed 2026102421. Four-cell equal-weight macro is primary, with all four cell estimates separately reported (n=50,10,50,50). Conservative interpretation for tiny ISIG-English cell.
- Null: shuffle anchor pages **within each source/view cell** before comparing predicted top1, to check for degenerate source priors. Use seed 2026102422. No test-selected thresholds or extra reranker variants.
- Required integrity tests: Phase18/19 source issue disjointness; 160 distinct issues; 480 query records; 3gram exclusion for keyword; all source anchors confirmed inside single raw paragraph; no zero-length clean/noisy queries; no class used as expert word label; multi-gold contains anchor; gold collection independent of rankings.

## Decision rule and publication boundary

If H20a passes but H20b fails, report query-mismatch diagnosis and reject universal lexical-reranking improvement. If H20b passes but gains are not consistent, do not claim source/view generality. Even if BOTH pass, **this does NOT establish method novelty, real OCR robustness, or independent human semantic retrieval**. The next gate is human-written information needs + independently adjudicated relevance, and authentic aligned scan OCR, followed by a competitive pretrained dense/hybrid benchmark, none of which may be claimed as completed here.
