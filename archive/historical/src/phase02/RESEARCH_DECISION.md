# Phase 2 research decision: adversarial archive-selection benchmark

## Decision: BENCHMARK GO; ORIGINAL-ALGORITHM CLAIM NOT JUSTIFIED

**Completed and independently verified.** A synthetic archival-selection benchmark with one finite historical corpus, six omission processes, both null and coherent injected lexical substitutions, source/page metadata, issue grouping, and pre-frozen held-out simulation draws. It shows a *difficult performance/identification trade-off*. It does **not** demonstrate an algorithmic discovery or historical changes of meaning.

### Scientific target and preparation

- Source: user's 1885 XhTxt pages in Imvo and Isigidimi. 292 pages, 61 issues, 366,550 alphabetic words in the original data. TRAIN 172 pages / 36 issues; DEV 52 / 11; TEST 68 / 14. Per-source test: 44 Imvo pages from 11 issues, 24 Isigidimi pages from three issues.
- Eligible word forms: 80 selected using TRAIN counts from Phase1's previously extracted 200-word pool, including 68 exact unchanged/null forms, six artificially increased and six artificially decreased forms. **The pool was originally selected using 1885 data, so TEST is not project-level untouched.**
- Two counterfactual periods are built from the **same finite set of 1885 pages**. One has unchanged counts; the other has coherent synthetic word-form substitutions in period B while keeping each page's total token count fixed. This is not a time-based semantic change and should never be described as such.
- Target: the difference in *uniform page means* of clipped per-page lexical rates (cap 0.02). Tables express effect/MAE in these rates times 10,000. This is **not** a word-volume-weighted frequency.
- Six selection mechanisms, 80 development and 160 held-out selection realizations per mechanism, each run on null and injected data (test: 1,920 complete simulated archive views, 80 word forms each).
- Every missing page's index, source, page position, issue and **word total** is assumed present in a privileged audit frame. In genuine survival-biased archives this information may itself be unknown; the benchmark is therefore optimistic about identifiable sampling weights and missing fractions.

### Key held-out results

For hidden-text outcome-dependent loss, injected world, averaged over 160 test selection masks:

| Method | MAE in rates ×10,000 | False sign on unchanged features | Correct sign on 12 injected features |
|---|---:|---:|---:|
| Pooled observed mean | 5.676 | 16.7% | 35.9% |
| Source/page standardized | 5.154 | 14.1% | 38.3% |
| Exact source/page-cell matching | 5.298 | 15.1% | 38.0% |
| KNOWN true propensity / oracle HT | 4.366 | 13.2% | 43.9% |
| Fitted propensity using visible frame | **3.684** | **9.8%** | 35.8% |

The point methods use a precommitted absolute decision threshold of 0.00075 (7.5/10,000); these sign decisions are **not** hypothesis-test discoveries, and these false-sign percentages are **not** nominal test Type-I errors. The true propensity estimator is non-deployable because the hidden lexical selection mechanism is not known to an analyst.

For the same hidden-text loss mechanism, set-valued estimators (injected world):

| Assumption | True effect within identification set | Fraction of signals correctly sign-resolved | Wrong sign on null forms | Assumption satisfied across features and selections |
|---|---:|---:|---:|---:|
| Only `Y in [0, .02]` (Manski bounds) | **100%** | 0.0% | 0.0% | Guaranteed by target definition + full frame |
| Missing–observed stratum mean gap <= 0.002 | 93.5% | 0.0% | 6.8% | 72.2% |
| Missing–observed stratum mean gap <= 0.0005 | **64.6%** | **66.6%** | **34.9%** | **2.6%** |

The restricted intervals are **assumption-conditional identification sets**, not 95% statistical confidence intervals. 'Effect contained' is the proportion of 80 finite target effects inside the interval across masks, including exact nulls and signals. The unrestricted bounds' 100% inclusion follows algebraically once missing fractions and clipping bounds are known; its value is its proof, not an empirical calibration breakthrough.

Additional key tests:

- **Whole-issue dropout:** source/page standardization still estimates 97.5% of draws and is often accurate, so issue loss need not imply an unsolved selection problem in this particular design. This is an important negative control.
- **Zero-support cell:** one Isigidimi page-half is completely unobserved in pseudo-period B. Standardization, matching, and valid known-propensity IPW abstain in 100% of draws; estimated-propensity regression still produces predictions by extrapolation, which cannot be interpreted as nonparametrically identified.
- **Repeated-text preferential selection:** with 196/292 historical pages containing a cross-issue repeated 13-token sequence, the simulator selects documents based on this latent property; ordinary observed-source adjustment reduces but does not eliminate estimation error.
- **MAR conditions:** useful standard baselines often lower error, and fitted-propensity weighting gives as little as 1.513 MAE under visible length/date selection. This benchmark does not support a claim that standard corrections always fail.
- **DEV to TEST:** the qualitative ordering (unknown-MNAR failure, bound-width trade-off, zero-cell impossibility) persists under fixed scenarios, but both partitions use previously examined 1885 issues. Repeated masks are not independent corpora.

### Interpretation and novelty gate

* There is a scientifically real **estimand/assumption/utility dilemma**, not evidence of a new computational method. When missing outcomes are unrestricted, one cannot determine the sign of most tiny lexical shifts from a partial archive. More confident results require additional **untestable or externally verified assumptions**. Changing the algorithm without adding justified information cannot defeat this identification limit.
* The literature already contains missing-data partial identification (Manski), confidence intervals for partially identified parameters (Imbens–Manski), source/poststratification and MNAR sensitivity/pattern-mixture methods. Our new research artefact is an archival-specific benchmark prototype, not these statistical ideas.
* The finite-population known-frame assumptions are unusually favorable. Missing issue counts, document word lengths, or whole source-years in real archives may make identification harder still.
* A 200-word candidate pool was chosen from the same 1885 data during Phase1; test outcomes and masks were newly simulated and frozen before execution, but this should not be marketed as an independent external validation.
* No article type, authorship, political ideology, historical cultural change or human impact is measured.

**Gate verdict:** Phase2 benchmark engineering/verification PASS. Publication-level novelty FAIL/NOT SHOWN. Advance to competitive statistical and semantic-change baseline replication ONLY if there is a path to independent source material and externally defensible missingness assumptions. Kill any proposed 'new method' that wins solely by assuming away the missing data or abstaining on every difficult case.

## Reproduction and claims

See `PRECOMMIT.md`, `ASSUMPTION_LEDGER.md`, both `outputs/*/summary_metrics.csv`, figures, code, and `outputs/*/INDEPENDENT_VECTOR_AUDIT.json`. The independent verifier retokenizes all 292 pages; reconstructs 2,073,600 point/set entries in DEV+TEST; replays 180,480 individual simulated observations; checks 192 summary rows and spot-checks learned-propensity and matching estimators. An entirely fresh extraction/replay is separately certified. The original historical TXT ZIP must be provided separately to reproduce raw-text tokenization.
