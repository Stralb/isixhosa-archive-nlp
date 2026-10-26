# ArchiveLens: calibration and attribution follow-up

**26 October 2026 | Post-hoc conditional follow-up to an already-inspected historical archive.**

## Research questions

1. Are ArchiveLens surface-pattern direction warnings unusual under a month-stratified publication-issue permutation benchmark, with multiplicity adjustment?
2. Are warnings sensitive to individual issues, exact repeated passages or page-position exposure?

## Experiment 1 — Permutation calibration

Exactly 4,999 within-calendar-month label permutations preserve whole publication issues and year/month issue counts. Pattern selection remains fixed from 1887, and reference years are 1885/1888. Both the year-effect test and the maximum sampling-estimand gap are corrected across 120 patterns per title with Holm FWER and BH FDR. The previously reported 3,000-replicate bootstraps are unchanged and unadjusted.

| Newspaper | Original sign warnings | Mean null warnings [central 95%] | Upper-tail p | Lower-tail p | Holm year effects | BH year effects | Holm sensitivity gaps | BH sensitivity gaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IMVO | **23/120** | 29.02 [19,40] | 0.8882 | 0.1582 | 26 | 54 | 29 | 58 |
| ISIG | **4/120** | 8.30 [3,16] | 0.9486 | 0.1192 | 0 | 0 | 0 | 0 |

The permutation distribution is a **reference benchmark**, not an unconditional estimate of false-alarm rate. Exchanging historical year labels can break real temporal structure, and 120 four-grams are correlated. The count test asks if the *number* of point-sign flags is unusually high or low; it does **not** calibrate all existing bootstrap support-loss warnings.

## Experiment 2 — Issue, exact repetition and page-position attribution

| Newspaper | Original flagged union | Flags with baseline year-effect sign changed by removing one issue | Exact 13-gram repeated-word share 1885 / 1888 | Flagged mass signs changed after masking repeated passages | Original reversal warnings disappearing after masking |
|---|---:|---:|---:|---:|---:|
| IMVO | **32** | **7** | 20.0% / 21.9% | 2 | 10 |
| ISIG | **6** | **4** | 2.4% / 1.3% | 1 | 1 |

Leave-one-out calculations remove entire publication issues and report the maximal shift per warning; these identify influential *sampling units*, not historical causes. Exact repeated-passage masking removes words covered by normalized 13-word sequences recurring across distinct issues **within the same newspaper**, then reruns all four estimands on the retained eligible views. Masked-text results change the eligible sample if pages fall below 40 remaining words, so they are a **sensitivity test**, not a controlled causal effect of duplication.

Page-position contribution tables use page numbers 1–4 and >=5. Position alone is not an annotated article type, newspaper topic, editor, OCR fidelity or masthead detection.

## Responsible NLP conclusion

ArchiveLens can expose when surface-pattern trends are sensitive to sampling unit, and its warnings can be checked against an explicit randomisation reference and traced to influential archive structure. A warning is **not** a linguistic or cultural discovery, a moral fairness score, or proof that a year trend is spurious. Multiple-comparison-adjusted permutation findings and unsupported results are reported rather than cherry-picked.

## Limitations

- The 1885/1888 source was examined repeatedly before this follow-up; there is no fresh external independent holdout.
- These patterns are within-word four-character fragments, not validated morphemes, sentiment, named entities or historical ideas.
- The two newspapers are in the same archive; year and newspaper source can co-vary with genre, preservation and editorial practices.
- Synthetic controls and permutation diagnostics do not guarantee calibration under arbitrary historical domain shifts.
- Holm controls a family of 120 tests per title and family under the reference null, but different null families were separately adjusted; BH statements require dependence caution.
- Existing bootstrap support-loss warnings are descriptive, not covered by the new permutation benchmark.
- Exact 13-token repeat masking is incomplete for paraphrases, short boilerplate, formatting artifacts and cross-source repeats.

## Reproduce

```bash
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify
```

The main evidence is in `results/archivelens/followup/`: `calibration.csv`, `attribution.csv`, `influential_issues.csv`, `duplicate_sensitivity.csv`, `duplicate_coverage.csv`, `page_positions.csv`, `summary.json`, plus figure outputs. The old ArchiveLens result files remain unchanged.
