# Phase 20: research decision (no paper drafting)

**STATUS: COMPLETE.** Frozen protocol and pre-scoring feasibility amendment are preserved with SHA256. All 462 new queries and BM25 scores, per-query rankings, uncertainty, and integrity tests are saved.

## Objective and design

Does query construction regime explain when full-page fuzzy reranking improves or harms known-page retrieval while holding source issue, source page, evidence anchor, index and candidate budget constant?

154 issues disjoint from all Phase18 balanced-query and Phase19 sampled source issues, with 3 query forms on each issue: (1) six spaced nonverbatim source-derived keywords, (2) clean five-word excerpt, (3) same excerpt with two interior character substitutions and one word split/join. Every query shares a 13-word within-paragraph evidence anchor, also matched to literal reprints. Four source/view cells: Imvo English 50, Isigidimi English 4, Imvo isiXhosa view 50, Isigidimi isiXhosa view 50. The originally planned ten Isigidimi English issues were infeasible under fixed eligibility; sample-size amendment was recorded before scoring. The tiny four-issue cell limits generalisation.

Seven fixed retrieval policies use equal top20 budgets: raw BM25; raw BM25 plus full-page RapidFuzz; language-gated English candidate augmentation plus fuzzy; unconditional English candidate augmentation plus fuzzy; hard language routing plus fuzzy; raw BM25 with within-120-word distinct-query-term coverage reranking; and gated candidates with the same local reranking. The local reranker was chosen on old Phase19 data only, not tuned after Phase20 evaluation. Issue-paired four-cell equal-weight bootstrap: 10,000 replicates, frozen seeds.

## Main source-anchor Top-1 results (equal four-cell macro)

| Policy | Six nonverbatim keywords | Clean copied five words | Synthetic noisy copied words |
|---|---:|---:|---:|
| Plain BM25 | **90.25%** | 60.25% | 18.75% |
| Full-page fuzzy reranking | 46.25% | **82.25%** | **44.25%** |
| Gated candidate pool + fuzzy | 46.25% | 82.75% | 44.75% |
| Local 120-word lexical reranking | 91.75% | 75.75% | 20.75% |

**H20a SUPPORTED:** The prespecified interaction `(fuzzy - BM25) on noisy - (fuzzy - BM25) on keywords` is **+69.5 pp**, 95% paired issue-bootstrap CI **[+55.5, +83.5]**. The three-cell sensitivity (omitting Isigidimi English n=4) is **+76.0 pp**, CI **[+66.7, +85.3]**. No claim of general semantic-query validity is justified.

**H20b REJECTED:** The previously selected local reranker improves keyword retrieval only **+1.5 pp**, CI [0.0, +3.5]. That fails the frozen >=5 pp with strictly-positive lower CI threshold, despite earlier Phase19 development gains. No validated novel retrieval method.

**Language-gated augmentation offers no meaningful improvement here:** +0.0 pp for keyword queries and +0.5 pp for clean and synthetic noisy excerpts relative to raw fuzzy.

## Why this happened: pipeline diagnostics (not a causal mediation proof)

On keyword lists, raw BM25 retrieves a valid 13-word-evidence page in its top20 for **99.5%** of tests. Full-page fuzzy changes 88 originally correct source-page Top-1 results to incorrect, recovering 0 missed anchors. This constructed query regime is primarily **ranking-limited**, not candidate-limited.

On noisy excerpts, raw candidate recall@20 is only **45.75%**. Fuzzy reranking repairs 28 missed anchor predictions, displacing zero previously correct raw-BM25 anchors, but it cannot recover a source page excluded from the top20 pool. This condition is primarily **candidate-coverage-limited**. Gated candidate augmentation moves recall@20 only to **46.25%**.

The keyword generator intentionally selects long/distinctive source words, which biases this proxy toward lexical BM25. Query length, term distinctiveness, and corruption are different across regimes; do not interpret the interaction as the isolated effect of natural human intent or OCR. Source-view membership is a silver field label, not expert word-level language annotation. Clean labels are originating pages plus exact 13-token reprints, not exhaustive semantic relevance. Isigidimi English n=4 is insufficient for broad statements. Repeated 13-word anchors occur for eight issues. No real OCR-aligned images, no human-written queries, and no pretrained multilingual dense/hybrid baseline were scored.

## Research decision

**ACCEPT** the controlled query-regime mismatch result. **REJECT** both a generalizable local-reranker claim and an algorithmic novelty claim for language-gated candidate enrichment. Preserve Phases 18 and 19 unchanged: their Top-1 tasks and relevance rules differ, so raw percentages must not be treated as directly equivalent.

**Next falsifiable gate:** collect *independent* historically meaningful search needs with independently assessed page-level relevance, and authentic OCR-to-corrected-text page alignments. Practical public lead: National Library of South Africa scanned/OCR page, Imvo Zabantsundu 1885-08-26 page 1, https://cdm21048.contentdm.oclc.org/digital/collection/p21048coll37/id/1477/. Matching supplied archive filename `IMVO_1885-08-26_p001.txt` exists. Text alignment, rights, and independence NOT YET VERIFIED; no OCR experiment can be claimed. Compare raw BM25, fuzzy, gated, hard/unconditional, character-tolerant retrieval and competitive dense/hybrid models under equal budgets. No paper or abstract until requested.
