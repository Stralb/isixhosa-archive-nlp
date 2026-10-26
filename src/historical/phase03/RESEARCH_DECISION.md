# Phase 3: competitive published-method inference — scientific decision

## Verdict
**PHASE 3 EMPIRICAL BASELINE AND REPRODUCIBILITY GATES PASS. A NEW ALGORITHM IS NOT JUSTIFIED.** Strong existing methods address observed selection effectively but cannot magically overcome unconstrained hidden-text-dependent preservation. Empirical test-year change in error is a warning against development-only calibration. No new historical semantic discovery is supported.

## Essential benchmark-validity correction
Our Phase 2 counterfactual worlds used exactly the same historical pages across both pseudo-periods. Unchanged word forms were therefore identical **on each individual page**. The otherwise legitimate paired-overlap shortcut finds all 12 injected directional signs and zero nonzero changes in 68 unchanged features for all six scenarios, for every one of 160 TEST masks with nonempty intersections. This is an artificial path to a perfect score unavailable in ordinary longitudinal historical inference. See `verification/phase2_paired_shortcut.csv`. Phase 2 is retained unmodified as a disclosure, not as a competitive main benchmark.

Phase 3 instead uses two different **actual publication years** per comparison and no shared issue or page between periods. Dev: 1884 -> 1885 (496 pages: 204 + 292). Test: 1887 -> 1888 (588 pages: 284 + 304). Excluded two incomplete 1887 Imvo issues. All 80 forms selected from development years only. A synthetic INJECTED alternative alters 12 later-year lexical forms but leaves the real across-year baseline variation intact. Consequently the NATURAL world is **not a zero-change world**. We measure recovery of finite digitized-archive lexical rates, not genuine sense or social change.

## Fixed held-out test results (1887 -> 1888), injected world

| Missingness | Point method | MAE (per 10k page words) | Wrong decisions among calls | Mean practical decision coverage |
|---|---|---:|---:|---:|
| Source/page MAR | Pooled | 3.879 | 11.6% | 29.6% |
| Source/page MAR | Source/page standardized | 1.481 | 1.3% | 23.7% |
| Source/page MAR | Estimated propensity | 1.418 | 0.6% | 24.0% |
| Source/page MAR | AIPW | 1.378 | 0.4% | 23.7% |
| Hidden lexical-outcome MNAR | Pooled | 5.571 | 28.9% | 26.3% |
| Hidden lexical-outcome MNAR | Standardized | 6.086 | 31.6% | 25.7% |
| Hidden lexical-outcome MNAR | Estimated propensity | 3.445 | 20.4% | 26.1% |
| Hidden lexical-outcome MNAR | AIPW | 3.403 | 20.7% | 26.1% |
| Hidden lexical-outcome MNAR | Known propensity HT (oracle) | 3.397 | 10.3% | 29.1% |
| Whole source/page cell absent | Standardized / valid HT | NOT IDENTIFIED | no issued calls | 0% |
| Whole source/page cell absent | AIPW extrapolation | 1.299 | 2.8% | 24.2% |

Here an "incorrect decision" includes asserting a direction when the actual digitized-archive rate difference is at most two per 10,000, or asserting the wrong sign for larger changes. Calls use the frozen absolute point-estimate threshold 7.5 per 10,000. These are **not** statistical Type-I error rates. AIPW extrapolation when a cell has no observations cannot be treated as assumption-free inference even if the realized error happens to be small.

## Equal decision-coverage comparison (precommitted)
For hidden lexical MNAR in the INJECTED world at **20% decision coverage**, wrong-call fractions are:

| Method | Dev 1884-1885 | Test 1887-1888 |
|---|---:|---:|
| Pooled | 0.70% | 27.81% |
| Standardized | 0.94% | 30.00% |
| Estimated-IPW | 0.55% | 21.46% |
| Ridge imputation | 0.00% | 20.21% |
| AIPW | 0.23% | 21.82% |
| Oracle HT | 4.53% | 9.27% |

This is an important negative transfer result: a method or calibration scheme that appears safe on one historical year pair can fail on another. Both comparisons involve different source-year compositions, and the synthetic mechanisms have fixed equations but different historical covariate/outcome distributions. The equal-coverage ranking is within each simulated replicate, a privileged *benchmark risk-coverage diagnostic*, not a readily deployed selection threshold with proven out-of-sample guarantees.

## Partial identification and sampling uncertainty are distinct
- Unrestricted Manski-style finite-population identification sets contain **100%** of 1887-1888 simulation truths by arithmetic construction, but resolved **0%** of sign decisions under all six mechanisms. This is guaranteed given full page inventory and Y in [0,.02], not a learned 95% CI.
- Under hidden-outcome MNAR, delta-pattern-mixture assumptions permit more decisions but are often false: GAMMA_0005 (max missing-versus-observed cell mean gap 5 per 10k) yields 48.9% sign decisions, 37.9% wrong among issued calls, 66.0% target containment, and only **4.8%** truth-verified cellwise premise satisfaction. GAMMA_0020 yields 10.8% decisions, 35.5% wrong among issued calls, 93.2% containment and 73.7% premise satisfaction. Neither is a validated confidence interval.
- Known-propensity HT 95% *selection-design* Wald CI includes the complete-archive truth **94.5%** of the time under hidden outcome selection, when the simulator discloses the inaccessible true propensities. It is **not** deployable on real MNAR archives. Approximate source/issue cluster CIs for practical estimators do not have unconditional nominal coverage under unknown outcome-dependent missingness.
- Missing a whole source/page cell breaks positivity and prevents nonparametric point identification, even if a fitted prediction happens to approximate the realized target. No estimator can infer its unseen outcomes without extra assumptions or an external information source.

## Novelty and publication gate
Existing methods (standardization, IPW, AIPW, ridge imputation, partial identification, delta sensitivity and selective risk-coverage evaluation) already explain the observed properties. The benchmarking exercise establishes no new theorem, no novel method and no true cultural/semantic change. **Do NOT advance directly to named algorithm development.** Phase 4 requires evidence that a new, observable source of missingness information exists (bibliographic inventory, parallel newspaper editions, preservation logs, verified text-quality indicators), or it should become an impossibility/benchmark-methodology project instead.

## Strict limitations
- The target is the mean clipped *per-page* count rate in digitized XhTxt views; it is not a rate of actual historical publication, semantic change, political attitudes or marginalized speaker representation.
- Full inventory of missing pages and their lengths is intentionally supplied, unlike most real archives. This is an optimistic missing-data setting.
- Issue counts are limited, especially Imvo in 1884 (9 issues); repeated masks are not independent historical collections.
- The 1887 and 1888 issues were used previously in separate NLP research, so this is newly pre-frozen phase-specific evaluation but not untouched external historical validation.
- Synthetic injection changes word-form counts, not human-labeled word meanings. All naturally observed words can change across years; the unmodified world is not a known zero-change null.
- Under zero support, fitted logistic propensities and ridge outcomes extrapolate; their numerical scores do not establish identification.
- Preliminary paired-page audit was retrospective; primary adjacent-year protocol was frozen before Phase 3 scoring. The verifier's zero-cell gamma arithmetic mistake was recorded and corrected separately; no benchmark score was modified.

## Sources (prior work, NOT inventions here)
- Robins, Rotnitzky & Zhao (1994), inverse probability weighting and augmented estimating equations, JASA: https://doi.org/10.1080/01621459.1994.10476818
- Imbens & Manski (2004), confidence intervals for partially identified parameters: https://doi.org/10.1111/j.1468-0262.2004.00555.x
- National Research Council (2010), pattern-mixture mean-gap sensitivity: https://www.ncbi.nlm.nih.gov/books/NBK209895/
- Lee (2009), sample-selection trimming under monotonicity: https://doi.org/10.1111/j.1467-937X.2009.00536.x . **Not applicable without the treatment/selection monotonicity assumptions in our two-newspaper chronology.**
- El-Yaniv & Wiener (2010), selective risk-coverage: https://www.jmlr.org/papers/v11/el-yaniv10a.html
