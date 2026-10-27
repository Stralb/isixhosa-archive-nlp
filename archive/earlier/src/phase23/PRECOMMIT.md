# Phase 23 — Prospective exact-reprint-aware retrieval evaluation

Status: PROSPECTIVELY FROZEN on 2026-10-24, **before creating query files, indexing new queries, scoring any new Phase23 queries, or inspecting any Phase23 retrieval rankings**. No Phase18-22 outcomes, queries, or hypotheses are retroactively changed. A pre-protocol feasibility scan using only raw/xh text (not retrieval scores) counted candidate exact 13-token reuse in unused issues: 54/59 Imvo and 7/7 Isigidimi issues had at least one repeated candidate under preliminary five-position rules. This is explicitly precommit-time feasibility information, not an experiment result.

## Research question

When historical newspaper text is reproduced across issues, how much does strict origin-page Top-1 scoring undervalue an already returned copy of the **same verbatim evidence**, compared with a conservative, fully source-independent exact-span multi-gold evaluation?

**Task distinction:** source identification (retrieve the requested specific issue/page) versus content evidence retrieval (retrieve any page containing the exact source evidence). Neither labels general semantic answers. No new retrieval method is proposed.

## Frozen data construction

- Search corpus remains exactly the supplied 3,608 raw historical newspaper pages. No new data injected into the index.
- Exclude as Phase23 **query origins** every issue appearing in the frozen Phase18 balanced query results, Phase19, Phase20, Phase21 and Phase22 origin sets. The index is not altered by this exclusion; all source pages must remain retrievable.
- Use **only xh-view paragraphs** for anchor selection (silver archive block views; no expert-language claim). Valid paragraph length 50–350 Unicode alphabetic words; normalize via lowercased Unicode letters plus internal straight/curly apostrophes, exactly matching the Phase21 helper. Raw-page paragraph boundaries `\n\s*\n` MUST contain the candidate exact contiguous 13-word sequence; do not join across boundaries.
- At each qualifying paragraph propose 13-word anchors beginning at five deterministic positions 0, 25%, 50%, 75%, last. Reject if fewer than eight distinct word forms or fewer than six words of length >=4. This is **not** selected from retrieval output.
- Find verbatim occurrences of each proposed anchor in the **full raw corpus**, paragraph bounded, using the same tokenizer. Gold is every page containing that exact 13-word sequence (verified after selection). A *repeated anchor* must occur in two or more distinct issue dates. A *unique anchor* has one matching raw page. Same-issue duplicates do not count as reprints; source page must always be among gold.
- Fixed issue quotas: **28 IMVO and 7 ISIG issues**, each supplying exactly one repeated and one unique anchor. Among eligible issues pick by ascending SHA256-derived deterministic key `P23|2026102423|issue|<source>|<issue>`; no manual replacement or result-dependent sampling. If an issue cannot supply one valid repeated/unique pair, advance deterministically to next eligible unused issue in that source. For each chosen issue pick the repeated anchor with minimum SHA256 key `P23|2026102423|repeat|<source_page>|<para_idx>|<pos>`; prefer an exact unique anchor from the same paragraph as repeated (then same page, then issue), breaking ties with SHA256 `P23|2026102423|unique|...`. Enforce distinct anchors and pages/issue labels without user-created language annotations.
- Each anchor gets two deterministic query forms: (1) `literal7`, seven consecutive anchor words from the window maximizing unique word types, with ties centered first then by start index; (2) `keyword6`, six distinct alphabetic words of length >=4 chosen by descending word length then lexicographic tie-break, alphabetically re-ordered so they are not an original three-word string. Both are **source-derived surface-form proxies**, NOT independent human queries. Require at least 4 unique words in literal7. If a candidate fails to produce forms, it is ineligible before issue selection.
- Save query rows, chosen span labels, all exact gold pages, SHA256 checksums, source/issue eligibility and any skipped cases **before retrieval scoring**. Planned total **70 evidence units and 140 queries**. Only one pair per issue; 35 independent issue clusters for bootstrap. If 7 Isigidimi issues cannot satisfy quotas, STOP and declare feasibility failure without lowering conditions after seeing results.

## Frozen methods

1. Raw word BM25: sklearn CountVectorizer `token_pattern=(?u)\b[^\W\d_]+\b`, k1=1.2, b=.75, positive-scored top 20 only, score=0 if no shared vocab. Same as Phase21.
2. Raw BM25 -> whole-page fuzzy rerank@20: RapidFuzz partial_ratio on lowercase alphanumeric-stripped query and page, tie broken by original BM25 score and page order. Fuzzy only reranks; never introduces new candidates.
3. Archive xh-view BM25 -> fuzzy@20: same 20-candidate budget and fuzzy. Silver-view field only, no expert language inference.

Record at rank 1,5,20: (a) strict source-page correctness; (b) **exact passage equivalence** multi-gold correctness. Store ordered candidate IDs and scores for independent reproducibility.

## Prospective hypotheses, tests and decision rules

- **H23a (evaluation mismeasurement on copies):** for duplicated evidence in nonverbatim `keyword6` searches, **more than half** of strict-origin Top-1 misses of raw BM25 are nevertheless exact-evidence Top-1 successes. Metric = number of such recovered misses / all BM25 strict-origin misses in repeated category. Gate: at least ten strict-origin misses and observed fraction >=0.50. Separately report numerator/denominator and issue-bootstrap descriptive 95% interval; this is not a general population probability because duplicate-positive anchors are deliberately oversampled.
- **H23b (policy-comparison sensitivity):** on the repeated `keyword6` subset, changing gold from strict-origin to exact multi-gold changes the estimated Top-1 gap (BM25+fuzzy minus plain BM25) by **at least five percentage points in absolute value**. Report signed difference-in-differences and issue-bootstrap 95% interval. Passing threshold alone is descriptive; to claim a robust direction require CI exclude 0.
- **H23c (paired uniqueness control):** unique 13-word anchors must yield identical strict-origin and exact-equivalent scores for every query/policy, by construction. Any disagreement fails verification and invalidates the run.

For all methods/forms/types report count, source-specific and equal-newspaper macro Top1, Top5, recall20, origin/equivalence gaps, and all paired changes. Bootstrap **issues, not queries**, separately in source strata with 10,000 resamples and seed 2026102423; sample all selected issues within each source, retaining their two anchors and two forms together. Confidence intervals are repeated-issue/sample uncertainty only; do not generalize beyond source-derived query design.

## Threats, stop rules and negative controls

- Duplicate selection is a deliberately enriched comparison, not a census or prevalence estimate. Exact textual reuse does not prove same context, historic provenance, editorial intent or answer equivalence for a dated query. The query forms omit dates, so only **content/evidence retrieval** can credit a reprint; origin metrics remain appropriate for issue-specific questions.
- Our exact 13-word rule is high precision but low recall: reprints with spelling changes, OCR errors, broken paragraph boundaries or small edits are excluded. This gives a conservative, surface-identical equivalence set, NOT expert judgments or an exhaustive semantic relevance assessment.
- Check that all repeated anchors have cross-issue occurrence and all unique anchors one page, source page membership, no previously used origin issue, and no boundary-crossing anchor; save independent verifier.
- No method tuning, no post-result anchor modification, no paper drafting, no claim of novel retrieval algorithm, no claim of raw OCR (the NLSA pilot remains transcript-provenance-uncertain).
- After main test, optionally apply exactly the frozen matching rule to Phase22 external/clean pairs as **retrospective sensitivity only** (eight selected items, do not treat as Phase23 confirmation). Distinguish strict exact-span result from earlier post-hoc >=95 fuzzy-match diagnosis.

**Decision after test:** if H23a holds, document a measurable benchmark-label confound and recommend separating provenance vs content metrics; if not, reject the claim that exact copies commonly explain misses in this selected new holdout. Regardless, authenticity of machine OCR and genuinely independent information needs remain open, and no new method is claimed.
