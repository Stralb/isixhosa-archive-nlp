# Phase 22 — research decision / audit checkpoint

**Status:** Completed, SHA-frozen external-transcript pilot, independent verifier PASS. **No new retrieval algorithm validated; no paper or abstract drafted.**

## Research question and data

Can results based on artificial character corruption be safely transferred to historical text from an independent digitisation channel? This pilot tests a more limited, falsifiable version: whether a publicly available *page-linked transcript excerpt* retrieves its originating corrected archive page as well as a paired corrected excerpt.

Eight short NLSA ContentDM excerpts were frozen (6 Imvo, 2 Isigidimi), each matched by newspaper/date/page identifier to supplied IsiXIT corrected/transcribed pages. All 3,608 corrected newspaper pages remain in the searchable index. Each excerpt was matched to the closest equal-length-ish contiguous window *within its known source page* before scoring; it was never used to select a retrieval policy. Only 2 of 8 source issues had not been used as query origins in Phases 18–21. Five other records were in the previous benchmark issue-origin set, one in Phase 19 (six overlaps total). Thus the sample is independent in *transcription provenance*, not predominantly independent in issue origin.

The NLSA interface calls the material **Transcript**. It does not establish whether this is raw uncorrected OCR, corrected OCR, or a combination. We did **not** download/view/align full scanned images or verify production settings, and we cannot honestly call this a validated raw OCR benchmark. Source excerpt selections were opportunity-based and required some alignability to the corrected archive; the sample is not a random cross-section of historical OCR quality.

## Prespecified strict originating-page results (n=8)

| Method | External Top-1 | Corrected Top-1 | External origin R@20 | Corrected origin R@20 |
|---|---:|---:|---:|---:|
| Plain word BM25 | 4/8 | 5/8 | 6/8 | 8/8 |
| Word BM25 -> whole-page fuzzy@20 | 5/8 | 5/8 | 6/8 | 8/8 |
| Within-word char3 TF-IDF | 2/8 | 1/8 | 4/8 | 4/8 |
| Char3 -> whole-page fuzzy@20 | 3/8 | 3/8 | 4/8 | 4/8 |
| Archive-xh BM25 -> fuzzy@20 | 5/8 | 5/8 | 6/8 | 8/8 |

- BM25 external–corrected strict-origin candidate recall was −25.0 percentage points, issue-bootstrap *descriptive* 95% interval [−62.5, 0.0]; Top-1 difference −12.5 pp [−37.5, 0.0]. Bootstrap intervals on eight convenience-selected pages are not population-level confidence statements.
- Fuzzy reranking restored one external page to exact originating-page Top-1, keeping final Top-1 equal to corrected-query Top-1 (5/8 vs 5/8). It could not insert an absent origin page into its 20-candidate pool.
- The same char3 index that failed Phase 21 performs worse than word BM25 on this sample. No retuning or architectural substitution occurred.
- Median within-source textual alignment was approximately **99.39/100**, heavily shaped by selecting *alignable snippets*. One 1870 scan-linked header aligns more poorly (77.65/100). Relative text differences in these snippets are not validated character error rates.

## Central validity finding — repeated passages change what counts as a miss

**POST-HOC diagnostic, NOT preregistered:** On all **four** pages where raw BM25 fails the strict originating-page Top-1 check, the predicted alternate page contains a **near-identical** version of the corrected query (>=95% normalized partial string similarity). The same is true of the **three** fuzzy-BM25 strict-origin misses. For two very repetitive ad snippets, the exact normalized corrected query occurs in at least 9 or 22 archive pages, respectively. The threshold was applied after inspecting results and is not a human relevance label.

Therefore a number such as “4/8 Top-1” here measures **edition/page identification** for a short snippet, not necessarily the ability to recover the textual content a historical researcher seeks. In this pilot, candidate loss of the exact source page on two external snippets is confounded with competing nearly identical advertisements. **We cannot attribute the losses causally to external transcript corruption.** This is a stronger *evaluation validity caveat* than merely reporting a drop in strict page-ID accuracy.

The particular 1892-08-18 Imvo metadata has a printed issue-number discrepancy (the visible NLSA transcript reads No. 408 whereas our corrected archive page reads No. 407) despite matching dates; a second transcription or scan check would be necessary to decide which is accurate. It does not change the fixed page-ID calculation.

## Decisions

1. **Pass only a narrow external-source feasibility gate:** NLSA exposes accessible page-linked transcript text with matching filename/date/page references in both historical collections.
2. **Do not generalize or call this real OCR validation:** authentic uncorrected OCR provenance, scan/content alignment at scale, rights-controlled retrieval, and representative sampling remain unestablished.
3. **Do not claim a novel retrieval method:** no evidence supports adopting char3, language gating, or fuzzy reranking as a general superior method from these eight cases.
4. **Flag originating-page metrics as potentially inadequate:** repeated historical adverts and reprints imply evaluation should allow independently adjudicated answer-bearing passages or formally clustered duplicate-evidence units. Do not retroactively change Phases 18–22 gold labels.
5. **Next experiment:** obtain a licensed export of raw OCR + scan identifiers aligned to IsiXIT pages; independently collect information needs and blind relevance judgments, with explicit near-duplicate clusters and competitive word BM25/phrase-aware fuzzy/character and multilingual dense baselines. If this evidence is inaccessible, stop claiming OCR or semantic-query validity rather than extending synthetic proxies indefinitely.

## Data/legal boundary

The package contains eight short attributed NLSA transcript excerpts (<=24 alphabetic words per page), original URLs and hashes, **no full NLSA transcript exports or scans**, and no redistributed raw source archive. Permission/terms for any future larger OCR or scanned-data release need separate verification. Reader access to the original two IsiXIT source ZIPs or the original full handoff (for `pages.jsonl`) is required to execute retrieval from scratch.

## Reproducibility

- `PRECOMMIT.md` and its SHA256 were saved before any Phase 22 scoring.
- `queries_external_frozen.jsonl` and SHA256 were saved before clean passage construction or scoring.
- `paired_queries_frozen.jsonl` and SHA256 were saved before indexing/scoring.
- `run_phase22.py` writes BM25/char/field scores, per-query rank lists, origin-retrieval metrics and descriptive page bootstrap.
- `verify_phase22.py` independently verifies source metadata matches, clean passages, all SHA hashes, the saved 80 query-method decisions and all summary metrics; PASS.
- `posthoc_reprints.py` is explicitly retrospective and must not be presented as preregistered evidence.
