# Phase 24 | Prospective, UNENRICHED reprint-prevalence and metric-sensitivity study

Status: FROZEN 2026-10-24, before Phase24 anchor selection, exact-duplicate outcome inspection, or retrieval scoring. This study is motivated by the artificially 50% repeat-enriched Phase23 evaluation. Prior studies and their results are unchanged.

## Question

When evidence spans are sampled WITHOUT conditioning on whether they are reprinted, how frequently are exact 13-word paragraphs reused across historical newspaper issues, and does such reuse materially change evaluation or the BM25-versus-fuzzy comparison? The task is exact-text evidence recovery, distinct from retrieval of a particular newspaper edition. This is NOT a test of semantic relevance to human-written information needs.

## Feasibility-known, not outcome-known

The supplied corpus contains 3,608 raw pages and 718 identifiable issues (517 Imvo, 201 Isigidimi). A text-only feasibility scan found at least one 50-350-word raw paragraph in every issue; it did not inspect cross-issue repetition or retrieval rankings. The Phase18-23 query-origin union leaves 31 unused Imvo issues and ZERO unused Isigidimi issues. Consequently the archive-wide study is DESCRIPTIVE / prospective new-anchor validation, not a balanced independent-issue replication. Independently unused Imvo issues are a secondary pre-specified sensitivity analysis.

## Sampling units and fixed random choices

1. For EVERY issue, draw one paragraph-uniform anchor (frame A) and one eligible-token-position-weighted anchor (frame B). Sampling uses SHA256-derived independent per-issue RNG seeds fixed by the literal `2026102424|<issue>|<frame>`.
2. Raw newspaper pages are the sampling source; archive XhTxt/EnTxt flags are NOT labels or selection gates. Eligible paragraphs contain 50-350 surface-form tokens under regex Unicode letters with internal apostrophes; no crossing paragraph boundaries.
3. Anchor = 13 contiguous words. Eligible anchors have >=8 unique surface forms and >=6 distinct alphabetic forms of >=4 characters; construct a six-keyword proxy from the six longest distinct alphabetic words, alphabetically sorted. Reject a proxy if it retains ANY anchor three-token sequence. Sampling is uniform over the eligible candidate windows in the specified frame; where rejection sampling is used, the final conditional distribution must be verified. Exclude an issue from a frame only if there are truly no eligible anchors (report how many).
4. Frame A: select a page uniformly from all pages with eligible paragraphs in the issue; a paragraph uniformly within selected page; an eligible anchor start uniformly in that paragraph. Frame B: select uniformly among all eligible anchor-start positions across issue paragraphs, so long text contributes proportionally more opportunities. Sampling uses only text feasibility, never duplicate labels or BM25 scores. Anchors are not deliberately selected to be repeated or unique. If the two frames happen to produce the same anchor, keep both and cluster by issue.
5. A full scan of ALL 3,608 raw pages then constructs exact 13-token match sets, restricted to individual raw paragraph boundaries. Gold sets include all pages with the exact anchor; `cross_issue_repeat` requires >=2 distinct issue IDs. Record within-issue multi-page duplicates separately. Do not infer context, genre, semantic equivalence or authorship from text identity.
6. One fixed six-keyword nonverbatim query per anchor; no clean or synthetic corrupted-query condition. Index all original 3,608 raw pages; no data are removed from the index to protect origins. Query-origin overlap with phases 18-23 must be explicitly disclosed, not called independent testing. Do not hand edit sampled queries or drop poor-performing ones.

## Methods / frozen evaluation

- Raw-word BM25 using precisely the Phase23 `CountVectorizer` word regex, k1=1.2, b=0.75, log1p BM25 IDF, stable ties. No tuned parameters.
- BM25+full-page fuzzy reranking only within its strictly positive top-20 BM25 pool, `rapidfuzz.partial_ratio` of normalized query against normalized entire raw page, original BM25 as tie-breaker, exactly Phase23.
- For every query report strict source-page Top-1, any exact-text evidence page Top-1, strict and evidence Recall@20, cross-issue exact duplicate indicator. A Top-1 content success does NOT imply correct page provenance.
- Report separately by source/frame with equal-source macro; pooled numbers only alongside source-stratified numbers. Paired 10,000-replicate issue bootstrap, fixed seed `2026102424`; paired resampling preserves the two frames/queries within each issue. Descriptive confidence intervals, not random population inference beyond the sampled archive/protocol.
- Natural prevalence will be measured over all eligible issues; no enrichment by duplicate status. Pre-specified sensitivity: compare prevalence/impact among the 31 previously unused Imvo issue origins. Do not use this subset to tune the rule.

## Prospective falsifiable claims and gates

H24a (REPRINT PREVALENCE): Under uniform-issue sampling, the equal-source macro fraction with exact cross-ISSUE repeat is >= 5%, and issue-bootstrap lower 95% bound is > 2%. Test A and B separately; classify supported only if BOTH frames meet both thresholds. If not, classify NOT SUPPORTED or INCONCLUSIVE (if point >=5% but interval crosses 2%). This is not an estimate of the frequency of historians' queries.

H24b (STRICT-LABEL UNDERCOUNT): The equal-source macro difference (BM25 evidence Top-1 minus source-page Top-1) is >= +5 percentage points and the issue-bootstrap lower 95% bound > 0, for BOTH frames. If not, the operational-materiality claim is NOT SUPPORTED/INCONCLUSIVE, even if some individual duplicate cases show large gaps.

H24c (MODEL-COMPARISON SENSITIVITY): On the unconditioned queries, the contrast [(fuzzy - BM25) under evidence Top-1] minus [(fuzzy - BM25) under origin Top-1] has absolute magnitude >= 5 percentage points AND a paired source-stratified issue-bootstrap 95% interval strictly outside zero on BOTH frames. If not, material benchmark ranking changes under natural anchor sampling are NOT SUPPORTED/INCONCLUSIVE.

Avoid treating H24a/b/c as three independent discovery p-values; the endpoints are related and thresholds are practical predefined gates rather than hypothesis-testing significance levels.

## Integrity / leakage / ethics

- SHA256 freeze this protocol before building anchors; freeze anchor/audit data before computing or inspecting model rankings. Independent verification will reproduce source-page coverage, sampled eligibility, anchor gold sets, all BM25 scores and metric summaries (or report verification failure honestly).
- Unicode surface-form tokens are approximate; not linguistic segmentation or modern isiXhosa labels. No isiXhosa expert annotations are invented.
- Raw page collection contains repeated notices, advertising, and historical stereotypes; exact matches alone cannot determine historical meaning. This study cannot support claims about culture, fairness outcomes, real OCR, or independent user semantics.
- Original raw texts belong to archive sources; portable package will contain small query anchors/summaries but NOT redistribute the 101MB archive.
- No paper drafting. No newly tuned retriever. If negative, reject over-generalization of Phase23 prevalence and prioritize human queries, OCR provenance, and relevance adjudication.
