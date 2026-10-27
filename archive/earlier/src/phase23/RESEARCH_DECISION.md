# Phase 23 — Research decision checkpoint (2026-10-24)

**Status: PRECOMMITTED REPRINT-AWARE EVALUATION COMPLETE; independent full numerical verification PASSED.** Frozen Phase18–22 results remain untouched. No new retrieval algorithm, independent human query benchmark, raw-OCR benchmark, or paper/abstract is claimed.

## Objective

Test whether a newspaper search can find the **same verbatim historical evidence on a different issue/page**, yet be marked wrong because its benchmark gold contains only the original page ID. Keep **provenance / specific-edition retrieval** distinct from **textual evidence retrieval**.

## What was actually frozen

- SHA256-frozen protocol `PRECOMMIT.md` before Phase23 query construction or ranking (`1aa866c9e6310449012b7dbc6cd54a93726338ac05dd1c1e6fd57c0783676f6e`). Preliminary text-only feasibility looked at where exact repeated spans existed, not model outcomes.
- **3,608 archived raw pages**, same unchanged searchable index, 28 unused Imvo issue origins and 7 unused Isigidimi issue origins, held out as query origins from **all Phases 18–22**.
- One exactly repeated 13-word span and one unique 13-word control per issue, both verified **within individual raw paragraph boundaries**. A repeated span must appear in **at least two distinct newspaper issues**. One normalized exact textual span defines every multi-gold copy. The original page always remains a positive gold.
- Two queries per anchor: a literal 7-word excerpt and reordered six-keyword proxy. **70 anchors, 140 source-derived queries**. Repeated copy multiplicity ranges 2–75 pages. These are deliberately duplicated/unique balanced samples, NOT an estimate of naturally encountered reprint prevalence.
- Methods: untouched full raw BM25, raw BM25+whole-page-fuzzy rerank of the same top-20 candidates, and archive-xh-field BM25+fuzzy@20. No candidate-budget or threshold tuning.
- H23a: at least 10 strict-origin BM25 mistakes, >=50% of which retrieve the same exact anchor on another page. H23b: original-page vs exact-multi-gold scoring changes fuzzy-vs-BM25 ranking gap by >=5 points (direction robust only if source-stratified issue-bootstrap interval excludes 0). H23c: unique anchors must produce zero evaluation-gap under both definitions.

## Confirmatory results (equal newspaper macro unless counts shown)

### Nonverbatim keyword queries, repeated evidence

| Method | Origin Top-1 | Exact-evidence Top-1 | Origin Top-5 | Exact-evidence Top-5 | Origin R@20 | Exact-evidence R@20 |
|---|---:|---:|---:|---:|---:|---:|
| Plain BM25 | 28.6% | **73.2%** | 51.8% | 92.9% | 91.1% | 96.4% |
| BM25 + fuzzy@20 | 28.6% | 58.9% | 39.3% | 83.9% | 91.1% | 96.4% |
| Xh BM25 + fuzzy@20 | 28.6% | 58.9% | 39.3% | 83.9% | 92.9% | 96.4% |

- **H23a PASSED:** `19 / 28 = 67.9%` of pooled BM25 strict-origin Top-1 misses return an exact copy of the 13-token evidence; source-stratified issue-bootstrap descriptive 95% interval **[50.0%, 84.6%]**. The event is a **wrong originating page / right verbatim passage** under a content-search interpretation, not a correct edition identification.
- **H23b PASSED:** fuzzy-minus-plain BM25 Top-1 appears tied at **0.0 pp** under strict origin labels but changes to **−14.3 pp** under exact evidence labels, a difference-in-differences of **−14.3 pp**, paired issue bootstrap **[−30.4, −1.8] pp**. The effect direction is negative for **both** newspapers (−14.3 pp each). This is a model-comparison sensitivity to gold construction, NOT a methodological innovation.
- Source-specific raw-BM25 exact minus strict Top-1 gaps: **+60.7 pp Imvo, +28.6 pp Isigidimi**, equal-source macro **+44.6 pp**, paired issue-bootstrap **[+26.8, +64.3] pp**.

### Other frozen checks

- Repeated 7-word literal excerpt: plain BM25 improves from **17.9% origin** to **75.0% exact-evidence Top-1**; fuzzy from **17.9%** to **76.8%**. Thus the equivalence-gap phenomenon is not dependent on the six-keyword permutation, although magnitudes vary.
- Unique 13-word controls, nonverbatim keyword: plain BM25 **69.6%**, fuzzy BM25 **41.1%**, archive Xh-field fuzzy **46.4%**, identical under both relevance definitions.
- **H23c PASS, by construction:** all 210 unique-query-method evaluations have the same source-page and exact-gold Top-1/Top-5/R@20. This is an integrity check, not independent evidence of effectiveness.
- On the primary repeated keywords, plain BM25 already placed a matching evidence page within top 20 in **96.4%** of cases (macro). Strict origin R@20 was **91.1%**. The central issue here is whether a ranking is interpreted as recovering evidence or identifying a particular edition, not candidate retrieval throughput.

## Mechanistic/interpretive boundaries

1. **An exact passage reproduced elsewhere may be valid for retrieving its text, but not necessarily its original historical edition or meaning in context.** Users asking *what appeared in the 1889-10-10 edition?* still need strict provenance labels. Historical research may demand both edition-specific and evidence-level outcomes.
2. Strong prevalence enrichment. We selected one copied span and one unique span per issue deliberately. The 73.2% evidence Top-1 is **not** accuracy for naturally distributed historian questions or an archive-wide estimate; it is performance on this deliberately constructed test.
3. All prompts are created from supplied text, not independently authored user information needs, and no relevance was judged by an isiXhosa specialist. `XhTxt` is an imperfect archive-supplied block view. Exact 13-word matches are conservative textual duplicates, **not exhaustive semantic gold**.
4. Some repeated spans are recognizably advertisement/notice-like, and one appears on 75 pages. We have not genre-annotated the full sample and cannot claim the effect holds for non-repeated investigative articles or editorial content. The selected issues come from different years and newspaper contexts.
5. Isigidimi has only **seven unused eligible issue origins**, so even source-stratified uncertainty is sample-bounded. Wider transfer requires new independent issues, real historical text queries, and/or human judgements.
6. No raw OCR is established. The NLSA interface labeled Phase22 content **Transcript** without verified machine OCR. This phase uses the supplied corrected archive, not original scan OCR.

## Post-result sensitivity (exploratory; NOT pre-registered)

For the 27 repeated keyword anchors with **10 or fewer** exact-copy pages: pooled BM25 origin Top-1 **22.2%**; exact evidence Top-1 **70.4%**; fuzzy exact evidence **51.9%**. Of **21** strict-origin BM25 misses, **13** still retrieve identical text. This direction suggests the primary result is not driven solely by the very largest advertisement clusters, but the cutoff was chosen after viewing results and is **not confirmation**.

Retrospective application of the exact 13-word rule to Phase22 external transcripts finds an eligible exact-window reference on only **3/8** cleaned excerpts (five contain no 13-word window satisfying the frozen contiguous/paragraph rule). On those three, source-identification and exact-text hit counts do not differ. **Do not use Phase22 as confirmatory external replication of Phase23.** The earlier >=95% character-similarity diagnostic is a DIFFERENT, exploratory, less conservative relevance proxy.

## Independent verification

`verify_phase23.py`:
- reproduces issue-origin exclusions and checks no Phase18–22 overlap;
- independently reconstructs exact 13-token full-corpus gold and verifies their paragraph-contained source occurrences;
- checks hashes for the frozen precommit, query file and gold-audit file;
- re-indexes BOTH raw and Xh views, recomputing all **140 × 3,608** BM25 scores with **max difference exactly 0.0**;
- recomputes all **140 × 3** candidate rankings and page-level source/evidence metric flags, and checks the three frozen hypothesis results.

**Result: PASS.** The script preserves train/test separation insofar as this retrieval study does not train a model; the issue split distinguishes previously evaluated query origins from previously unused query origins.

## Scientific decision

**Accept a narrow, useful evaluation validity finding:** exact historical reprints can make single-origin-page retrieval labels severely underestimate recovery of verbatim evidence and can alter baseline rankings in a controlled, independent-issue test. This is not a novel retriever and not proof of human-search relevance. It is a candidate benchmark-quality contribution only if later supported by diverse independently written information needs and independently adjudicated context-sensitive relevance labels, particularly distinguishing syndicated/reprinted passages from provenance-critical search.

**Do not tune another lexicon/character/routing variant on the now-inspected Phase23 data.** Priorities: (1) obtain independent historian-informed queries and human-adjudicated page/passage labels, (2) collect verifiably raw machine OCR linked to exact scanned pages with rights/provenance checked, (3) compare against competitive dense and strong lexical baselines on truly independent evaluation. No paper drafted.
