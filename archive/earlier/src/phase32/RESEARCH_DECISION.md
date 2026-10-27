# Phase 32 — language visibility, publishing history and sampling choices
24 October 2026 · Completed computational experiment. No paper or abstract drafted.

## Question and provenance
Do historical newspaper sources and ordinary sampling units change which archive-assigned English/isiXhosa text an NLP system encounters?

Dataset of record: original 3,608 page-level split files and LanguageSplitTracker70-94.csv supplied by the user, historical editions 1870–1894. Labels `XhTxt` and `EnTxt` are assigned to copied blocks of archival text, NOT expert-confirmed word-level language IDs. All analyses apply to corrected digital versions of the newspapers, not raw newspaper OCR or true linguistic identity.

Pre-registered research decisions are in `PRECOMMIT.md`; SHA256 in `PRECOMMIT_SHA256.txt`. Sources were not modified. All page-level counts reproduce the tracker without discrepancy. 7,071,285 classified words and 21,288 remaining words in the raw-page tracker (0.3001%); one page has zero classified words and is excluded from ratios.

## Confirmatory tests

### H32a — matched newspaper contrast: SUPPORTED
Across 36 matched calendar year-month cells in 1885, 1887 and 1888, Imvo's issue-level English-view share exceeded Isigidimi's by **+28.491 percentage points** (fixed-month, within-month issue bootstrap 95% CI **[+27.408,+29.533] pp**, seed 3201). In a separate cell-bootstrap sensitivity allowing calendar-month variation, the interval was **[+26.019,+30.829] pp**. This is a source/document-composition contrast, not evidence that the institutions caused individuals to switch languages or an inference about political beliefs.

### H32b — sampling estimand: SUPPORTED
For Imvo in the same overlapping years, archive-English share is **29.475%** by classified word mass but **22.643%** averaging pages equally, a **+6.832 pp** difference (issue-cluster bootstrap 95% CI **[+5.891,+7.779] pp**, seed 3202). Average of issue shares is **29.432%**; do not conflate equal-issue with equal-page sampling. This compares two intentional dataset constructions, not estimator bias in an abstract sense: different sampling schemes target different populations. Previous Phase 2 exploratory work already noted a sampling-unit effect; this is a time-restricted, uncertainty-estimated check, not an independent discovery of the weighting principle.

### H32c — historically substantial Imvo change: NOT SUPPORTED
The issue-weighted English-view share was **29.415%** in 1885–1888 and **25.885%** in 1889–1894: delta **−3.529 pp**, with issue-bootstrap CI **[−4.892,−2.198] pp**, seed 3203. Although a modest decrease is observed, its magnitude is smaller than the preregistered |5 pp| substantive-change threshold. A sensitivity equally weighting years gave **−3.606 pp** (year-bootstrap interval **[−7.606,−0.427] pp**); only 4 vs 6 publication years makes generalization highly uncertain.

## Supporting diagnostics (not separate confirmatory discoveries)
- Imvo: top 10% of pages by English-view count contain **43.73%** of its English-view word mass.
- Isigidimi: top 10% contain **87.93%** of its English-view word mass, but English-view text is sparse in this newspaper so concentration is unsurprising.
- Imvo's page 3 contains **75.05%** of all English-view word mass while containing **41.44%** of all classified word mass in that paper. This is an archive layout/source structure diagnostic; page numbers do not provide true article genres.
- Pre/post a historical publication milestone (Isigidimi 1870–1875 vs 1876–1888): issue mean share goes **4.27% → 0.91%**; bootstrap CI for difference **[−6.49,−1.10] pp**. IMPORTANT: this was tested AFTER seeing main yearly allocations and is **EXPLORATORY**. The archive includes missing issues (including all of 1886) and the two phases differ in many ways; we cannot attribute the change to independent publication or claim population-language evolution.
- Newspaper-overlap estimates are not independent of earlier projects or data. The latest stage is a better-controlled source-composition comparison, not new evidence about sentiment.

## Interpretation and responsible NLP significance
A multilingual historical newspaper can appear more or less English-bearing depending on page versus word sampling. This is a measurable property of dataset assembly, not an intrinsic characteristic of isiXhosa or a validated model harm. If an NLP pipeline uniformly samples pages or excludes particular page positions, it could train on a materially different historical language distribution from one that allocates exposure by text volume. Training effects, actual harms, and culturally situated meaning were NOT evaluated here.

The source compositions also differ markedly even in the same publication years. This does not isolate editorial policy, genre, readership, historical power relationships, or source-specific language tagging standards. It underscores the need to report estimands and data provenance in Responsible NLP analysis.

## Publication gate
This is a defensible **dataset curation / measurement** result appropriate for the course's Responsible NLP emphasis. It is not a novel NLP method, a history-of-ideas discovery, or proof of fairness harms. For a stronger publication, pair this with the earlier prospective BPE-composition evidence and downstream LM findings, while clearly separating corpus-level allocation evidence from causal tokenizer interventions. Further meaningful progress requires item-level annotated genre/topic data or independent external texts; adding arbitrary budgets or tokenizers is not justified.

## Reproducibility
- `run_phase32.py`: consumes the original supplied ZIP, writes all page/issue/year counts and 3,000-replicate bootstrap estimates.
- `verify_phase32.py`: independently checks every page against source tracker and all three point contrasts.
- `plot_phase32.py`: generates visualizations from saved numeric tables.
- `RESULTS.json` and `INDEPENDENT_VERIFICATION.json`: machine-readable results and audits.
- `FRESH_REPLAY_VERIFICATION.json`: fresh-run comparison, without redistributing underlying newspaper texts.
