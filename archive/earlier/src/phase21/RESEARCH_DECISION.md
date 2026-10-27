# Phase 21 research decision — 2026-10-24

**STATUS: COMPLETE / INDEPENDENT CODE-LEVEL VERIFICATION PASSED. NO PAPER DRAFTED.**

## Objective

Test whether an explicitly constructed full-corpus, word-contained character-trigram TF-IDF retrieval baseline rescues the candidate-coverage bottleneck that Phase 20 observed for five-word excerpts with synthetic character and boundary corruption. Evaluate an equal-budget 15-word-BM25/5-character hybrid and ensure the fused ranking does not destroy keyword retrieval.

## Pre-result protocol and data

The SHA256-frozen `PRECOMMIT.md` preceded new Phase 21 query construction and all retrieval ranking. Selected 30 Imvo and 30 Isigidimi issues exclusively from source issues never used as Phase 18/19/20 query origins; 300 paired query forms (60 issues times 5 forms) against the unchanged 3,608 raw-page corpus. Archive `xh` files are source-designated views, **not expertly annotated word-level language labels**. An exact 13-word span within one raw paragraph supplies the gold; five anchors have multiple exact reprint pages, which are all accepted. This is an *automatically constructed known-evidence task*; no human semantic retrieval judgments.

Methods (20 candidates max, fixed before test): word BM25, BM25+full-page fuzzy, padded within-word char-3gram TF-IDF, char+fuzzy, 15-word/5-char fusion RRF, hybrid+fuzzy, and xh-field BM25+fuzzy. Bootstrap across 30 independent source issues per newspaper, with equal newspaper weighting, 10,000 resamples and fixed seed. No model tuning or scoring-dependent method changes.

## Primary results — 2 substitutions plus 1 word-boundary change

| Policy | Candidate recall@20 | Top-1 exact known evidence |
|---|---:|---:|
| Word BM25 | 56.7% | 16.7% |
| Word BM25 + fuzzy | 56.7% | 53.3% |
| Character-trigram TF-IDF | 20.0% | 8.3% |
| Character-trigram + fuzzy | 20.0% | 18.3% |
| 15-word/5-char hybrid RRF | 51.7% | 18.3% |
| 15-word/5-char hybrid + fuzzy | 51.7% | 48.3% |
| Archive-xh-field BM25 + fuzzy | 63.3% | 60.0% |

### Frozen hypothesis outcomes

- **H21a REJECTED:** char20−word20 candidate recall@20 = **−36.7 percentage points**, paired issue-bootstrap 95% CI **[−50.0, −23.3]**; newspaper-specific effects Imvo −43.3 pp and Isigidimi −30.0 pp. Pre-registered improvement needed +10 pp with CI lower > 0.
- **H21b REJECTED:** hybrid+fuzzy−word+fuzzy Top-1 = **−5.0 pp**, CI **[−11.7, 0.0]**. Imvo −3.3 pp, Isigidimi −6.7 pp. Needed +5 pp, CI lower > 0, and no >5 pp source-specific loss.
- **H21c REJECTED:** hybrid RRF−word BM25 on nonverbatim keyword Top-1 = **−61.7 pp**, CI **[−71.7, −51.7]**. Imvo −86.7 pp, Isigidimi −36.7 pp. Needed no worse than −5 pp in either newspaper. Word BM25 itself achieved 98.3% on these deliberately distinctive source-derived keywords; that does not generalize to ordinary questions.

Across synthetic-noise severity, raw word BM25 candidate recall@20 declines from 96.7% clean to 93.3% one-edit, 56.7% two-edit+boundary and 35.0% three-edit+boundary. The char3 baseline is poorer than BM25 in all four. A conventional xh-field BM25+fuzzy control leads the selected Top-1 table at moderate corruption; that difference is an **exploratory** observation and NOT a validated new routing method.

## Why the negative result is informative (descriptive, not causal)

- In the moderate-noise condition, `char20` found two known-evidence pages missing from word BM25's Top-20 but excluded 24 pages that word BM25 had recovered. The hybrid fuzzy method recovered zero newly correct Top-1 hits relative to word+fuzzy and lost three.
- On keyword queries, standalone char3 retrieval placed an Isigidimi page first for **56/60** queries despite balanced 30/30 newspaper origins. Its selected page median raw text length was 6,760 characters versus 15,144 for word BM25. These are *observed ranking biases*, not an isolated proof that cosine normalization, newspaper length, or corpus style caused the failure.
- Unique-13-word-anchor sensitivity, restricted retrospectively to 55 non-reprint cases, retains the same negative direction: word recall 56.4%, char recall 20.0%, word+fuzzy Top-1 54.5%, hybrid+fuzzy Top-1 49.1%.
- Within-newspaper shuffled-evidence null control returned 0–5% Top-1 in the moderate-noise condition, supporting that normal evaluation was not a trivial collection-prior lookup.

## Validity boundaries

- **This rejects ONLY the precommitted full-page within-word char-trigram TF-IDF implementation.** It does not compare char n-gram BM25, better calibrated hybrid scorers, neural retrieval, modern multilingual representations or field-aware trained models. A three-gram cosine representation can dilute discriminative rare-word matching; this is a proposed explanation, not established mechanism.
- The 300 queries are **source-derived**; the corruption is **synthetic**. The available gold marks only exact anchor matches, not every page that may answer an independent question. This does not measure actual users' semantic relevance, authenticity of historic OCR, social fairness, or general retrieval quality.
- The two newspaper sets contain different genres, dates, and layout properties. Silver language fields are not expert ground truth. Thirty independent issue origins per newspaper give issue-grouped uncertainty only over this constructed benchmark.
- The full archive remained searchable; issue disjointness concerns query **origins**, not holding source documents out of the retrieval index, which would change the task.
- With these 60 issues now used, only 7 Isigidimi issues remain outside the Phase18/19/20/21 query-origin sets under the existing eligibility rule. Another internally sourced Isigidimi confirmation would be badly underpowered. Reusing prior evaluation issues after seeing their results is exploratory, not new independent confirmation.

## Research decision and next gap

**Do not adopt or claim novelty for the tested char3 index or the hybrid.** The result narrows the next research need: use an independent evaluation resource (real information needs, independently assessed answer-bearing pages, and/or scan OCR aligned with the corrected historical pages) rather than trying further unvalidated lexical variants against these now-exposed synthetic tests. Before scoring such an evaluation, predefine source time/issue partitions, label protocol, fair candidate budgets, and competitive BM25/character/modern dense or hybrid baselines.

A publicly indexed National Library of South Africa Imvo page offers a transcript and digitized newspaper object (1885-08-26 page 1; source filename exists in the archive), but trustworthy OCR-to-corrected alignment and rights/terms for a scored collection have not been established here. **No real historical OCR benchmark was executed in Phase 21.**

## Reproducibility artifacts

`PRECOMMIT.md`, hash, `construct_queries.py`, frozen queries and anchor audits with hashes, `run_phase21.py`, 300-query rankings/candidate pools/score arrays, result JSON, summary/hypothesis CSVs, source-bias and negative-control diagnostics, `verify_phase21.py`, `make_assets.py`, and research diagnostic figures. The independent verifier reconstructed within-paragraph gold from raw documents and checked all 300 query-pool/prediction metrics against archived scores, returning `PASS`.
