# Phase 22 prospective protocol: external transcript transfer pilot

Date: 2026-10-24. Frozen before selecting cleaned paired spans, constructing any retrieval index or inspecting any Phase 22 scores. Prior Phase 18–21 experiments and their conclusions are unchanged.

## Research uncertainty

Can retrieval results from source-derived synthetic character substitutions be treated as evidence about independently digitised, real historical newspaper text? We test the narrower tractable claim: does a genuinely EXTERNAL scan-linked *transcript snippet* retrieve the same corrected archive source page as the best matching clean transcription snippet? This is a **pilot on transcript-channel mismatch**, not a test on verified raw OCR, and not on independent human information needs.

## Fixed source selection and feasibility gate

Use only the eight externally surfaced transcript records, matched by newspaper / issue date / page identifier to existing archive pages (NLSA ContentDM, manually cross-checked before scoring):
- IMVO_1885-08-26_p001: p21048coll37/id/1477
- IMVO_1889-05-30_p001: p21048coll37/id/2921
- IMVO_1889-06-13_p001: p21048coll37/id/3103
- IMVO_1890-09-04_p003: p21048coll37/id/3450
- IMVO_1892-08-18_p001: p21048coll37/id/501
- IMVO_1893-06-14_p001: p21048coll37/id/268
- ISIG_1870-10-14_p001: p21048coll72/id/9
- ISIG_1882-06-01_p004: p21048coll72/id/963

Other external URLs lack exactly matched supplied page IDs or available transcripts and are out of scope. Five-digit year/day filename equivalence alone is not proof of scan-content equivalence: report concordance with corrected same-page text. Corpus search index includes all 3,608 raw-pages. Each page is 1 independent unit; treat overlapping prior-origin usage as known, not unseen issue holdout. Six IMVO, two ISIG. External metadata labels these *Transcript*, but transcription-production provenance and OCR rawness are unknown. Use no expert language labels.

## Query production and lock

Select exactly one short (~14–24 word) excerpt visible in the public site's transcript per listed page, using a substantive within-page passage (rather than search rankings) with at least some correspondence to its matching local page; include only a small research excerpt per URL, not the complete scan/transcript, and record a source link. Selection necessarily involves manual source review and may favor alignable passages. Do not access index scores, candidate ranks, or outcomes until `queries_external_frozen.jsonl` has been saved and SHA256 recorded.

For each external snippet generate paired *edited transcript* query automatically from the best matching contiguous window in the **identified correct local page only**, using case-insensitive alphabetic-token matching and RapidFuzz ratio to select a window of equal token count plus/minus three. This uses known page metadata, not corpus-wide ranking. Retain every preselected page, even if alignment is poor. Save the resulting alignment similarity; do not choose a different external excerpt in response to score. Queries must not include the page date/file name metadata except if actually in the excerpt. Gold = explicitly matched source page only. We cannot claim exhaustive relevance in other pages, especially syndicated/repeated advertisements.

## Frozen retrieval methods and budgets

1. `BM25`: full raw archive-page word BM25, sklearn CountVectorizer alphabetic tokens, k1=1.2,b=.75; score=0 if no shared terms.
2. `BM25_FUZZY20`: raw BM25 top 20 positive-score candidates, then RapidFuzz partial ratio on ASCII-alphanumeric-stripped lowercased whole query / page, preserving BM25 order on tie.
3. `CHAR3_TFIDF`: the exact Phase 21 document word-contained, `^word$` padded trigram TF-IDF/cosine index, top 20.
4. `CHAR3_FUZZY20`: the previous char3 candidate list reranked by identical fuzzy operation.
5. `XH_BM25_FUZZY20`: Phase 21 archive `xh` source-designated field BM25+fuzzy; field is imperfect, source-language mismatch on English text expected.

For each query/policy compute exact labeled page Top1, Top5, and candidate inclusion at Top20. For each issue pair compute `external - clean` differences (score or binary) and report median alignment similarity and publisher/per-page stratification. No method tuning, score normalization tuning, stopping based on results, or change to original query strings after execution. Bootstrap 95% intervals over eight page units with seed 2026102422 are **descriptive only**, not population inference. No formal rejection/support claims from n=8; 2 Isigidimi pages make newspaper subgroup comparisons illustrative.

## Decision rules

If OCR-like transcript divergence is documented AND Top1/recall differs from corrected text for any pages, conclude real transcript channel shift exists in this pilot; **do not** claim generic OCR retrieval effectiveness. If all remain identical, do not infer portability to OCR, only lack of observed effect in an opportunistic sample. Regardless of result, independent human information needs, full machine-OCR provenance, rights-controlled larger samples, and separate cross-domain confirmation remain open. No paper, no abstract, no method novelty claim.

## Reproducibility requirements

Save external excerpts with URL and page_id, frozen SHA256, clean-window extraction code, executable retriever, per-query ranks and scores, diagnostics, eligibility/overlap audit, independent verifier and research decision. Source snippets contain at most 24 alphabetic words from any one external page; do not distribute complete NLSA scans or transcript texts.
