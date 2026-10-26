# Phase 1 scientific decision (2026-10-24)

## Verification-level findings, NOT semantic changes

* The only source used was the user's historical newspaper ZIP (SHA256 in `outputs/dataset_manifest.json`). The pilot reads **1885 archive-assigned XhTxt only**: **292 pages**, 49 Imvo publication issues (196 pages, 277,525 alphabetic word tokens), 12 Isigidimi issues (96 pages, 89,025 tokens). It evaluates **200 common word forms** across both newspapers; 1887+ heldout data were **not evaluated**.
* Two randomized synthetic-selection mechanisms were assessed, each in 200 repeated selections of disjoint issue groups. **SOURCE** shifts issue source shares 8:2 vs 2:8. **PAGE** selects 4:2 vs 2:4 early/late page-half issues within *each* newspaper. The arbitrary pseudo-periods are drawn from one year and do not correspond to actual changes in historical meaning.
* SOURCE: naive iid-token two-proportion p-values + BH(q=.05) yielded mean 45.32 and median **45** apparent change discoveries / 200 tested word forms. Source-equal standardized issue-cluster normal test gave mean 6.405 (median 6); source+page-normal gave mean 2.55 (median 2). The fully design-aware **oracle randomization** test gave mean **0.025** (median 0), with a nonzero discovery in **2.5%** of pseudo-period selections.
* PAGE: naive iid-token+BH gave mean **6.92**, median **6**; source-equal cluster-normal gave mean .41 (median 0); source+page cluster-normal gave mean 1.81 (median 1); oracle randomization gave mean **.075** (median 0), with at least one discovery in **5.5%** of simulations.
* Mean across-simulation median word-form absolute changes are SOURCE: POOLED 1.921, SOURCE 1.572, SOURCE+PAGE 1.578 occurrences / 10k tokens; PAGE: POOLED 1.665, SOURCE 1.655, SOURCE+PAGE 1.730. Point-error improvement is NOT guaranteed by finer stratification: sparse issue cells can inflate sampling variability.
* IMPORTANT: the normal-approximation inferential results for SOURCE and SOURCE+PAGE are **not calibrated guarantees**: some strata contain only two sampled issues. Token-IID testing is deliberately naive and not an adequate publishable comparator. Repeated Monte Carlo realizations reuse the same 1885 archive, so they are not independent corpora. The oracle is only possible because we know the complete finite input archive and the exact random selection process.

## Honest novelty gate

1. **Artifact gate — PASS:** substantial apparent pseudo-temporal signal can arise solely through archival composition and pseudoreplication.
2. **Easy-solution gate — REJECT as novel:** strong existing design-aware randomization inference correctly handles the two *known selection* scenarios, and source/page standardization removes much of the artifact. Simply proposing 'archive-aware reweighting' is insufficient for a new NLP methods paper.
3. **Hard-gap gate — OPEN / NOT YET SHOWN:** it may still be valuable to benchmark **unknown, outcome-dependent or nonpositive archival loss** and to create calibrated *selective* inference under defensible sensitivity assumptions. That methodological advance is a hypothesis, not a verified result or an original algorithm.
4. **Publication gate — NOT MET:** no competitive method, independent corpus, semantic gold labels, or statistically guaranteed inference has been developed. Phase 1 only justifies carrying an exploratory *risk-focused* benchmark into Phase 2, subject to later kill criteria.

## Next decision

**Proceed conditionally to Phase 2 with a narrower task**: a benchmark of partially observed archival selection with separate masks for (a) covariate-observed shifts, (b) within-stratum outcome-dependent censorship/selection, (c) missing source-page cells, (d) date/issue-dependent dropout and repetition. Evaluate simple fully competitive baselines and known partial-ID bounds *before* designing a new algorithm. Ask an expert to validate any historical subject/genre/sense interpretation; none is currently available. Use at least one truly independent corpus for any publication claim.

Source-data, code, and analysis are reproducible; do not present this as discovered semantic change, cultural impact, or a novel published method.
