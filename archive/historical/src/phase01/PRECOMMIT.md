# Phase 1 preregistered feasibility protocol (frozen before stress-test outcomes)

Research question: Are apparent historical word-form trends manufactured by source/page composition changes, and to what extent do ordinary, stronger controls already remove them?

Date: 2026-10-24. Source years audited and base counts inspected, but no Phase-1 stress-test outcomes computed at the time of this freeze. Phase 1 is a **diagnostic pilot**, not a publication-ready benchmark and not an analysis of semantic change.

## Population/units, provenance, split

Use the user-supplied `TxtLangFiles1870-1894_TB.zip` only. Restrict Phase 1 to **XhTxt 1885** for Imvo (`IMVO`) and Isigidimi (`ISIG`); archive `XhTxt` is an archive-assigned text view, not verified native-language or article-genre annotation. 1885 offers 49 Imvo issues × 4 pages and 12 Isigidimi issues × 8 pages. Never look at 1887/1888 evaluation outcomes for designing the benchmark. No article-level subject/speaker labels.

Tokenization: lower-case via Python casefold and Unicode alphabetic word regex `[^\\W\\d_]+` (the literal Python regex is `[^\W\d_]+`), no lemmatization, spelling modernization, or semantic inference. Each issue is the `SOURCE_YYYY-MM-DD` unit; all sampled pages from an issue stay in one pseudo-period. Group pages by within-newspaper half: Imvo pages 1-2 versus 3-4, Isigidimi pages 1-4 versus 5-8. These are **layout strata**, not genres. Candidate 200 common wordforms: length >=4, >=20 total occurrences in each newspaper, ranked by total occurrences in all 1885 source text (stable lexicographic tie break). Candidate selection precedes simulation. No claims that words are political concepts or their uses reflect senses.

## Known-null simulation

200 independent seeded draws (global RNG `20261024`), two selection interventions, all issue dates sampled without replacement within each replicate. Pseudo-period A and B are arbitrary groups sampled from **the same calendar year**. The underlying source/layout-stratum distribution is taken to be stationary *in expectation under the sampling design*; differences of realized issue samples are random noise, not literal absence of genuine within-year heterogeneity. Reuse of 1885 issues across independent replicates is expected and makes replications **dependent as archival data evidence**.

Scenario **SOURCE**: period A = 8 Imvo + 2 Isigidimi distinct issues, B = 2 Imvo + 8 Isigidimi distinct issues. Include all XhTxt pages. This deliberately changes source composition.

Scenario **PAGE**: for each newspaper separately select 12 distinct issues, six for A and six for B. A includes ONLY early-half pages from four of its six issues and late-half pages from two; B early-half pages from two and late-half pages from four. Thus source issue counts are equal but layout is intentionally shifted, with at least two issues per source-layout cell per pseudo-period.

All issue assignments and page selections are saved with deterministic seeds. No test-data-informed scenario tweaking permitted. Check disjoint issue IDs across pseudo-periods in each replicate, all cell positivity in PAGE, and exact repeated simulation replay.

## Comparators (no proposed novel method in Phase 1)

`POOLED`: prevalence difference in all observed pseudo-period word tokens (naive archival trend). `SOURCE`: equal-weight Imvo/Isigidimi prevalence within each period, then difference. `SOURCE_PAGE`: equal weight in four source×page-half cells per period. The latter two are **standard existing standardization baselines**. Standardization is unbiased for a stratum-stationary population when the sample mechanism is independent of lexical outcomes given its recorded strata, subject to finite-sample limitations. It cannot generally correct missing-not-at-random omissions *within* a recorded stratum.

Primary diagnostics: median across the 200 features of |B-A| per 10,000 observed word tokens for each replicate; aggregate median over replicates; ratio against POOLED and fraction of replications where controlled error is smaller. Also report BH-q<0.05 discovery counts from (1) an explicitly overconfident independent-token z test, and (2) exploratory delta-method **issue-cluster** SE for the two stratified baselines. The naive token-z is intentionally invalid for correlated text, and cluster-normal p-values are only descriptive because some cells have just two issues. Do NOT claim formal familywise or population Type-I-error guarantees.

No hypothesis about universal superiority of a new model. **Gate A:** artifact exists: SOURCE and/or PAGE pseudo-time interventions create nontrivial differences for some common forms, reproducibly in this pilot. **Gate B:** does the appropriate simple source/page standardization substantially reduce apparent drift? If yes, the straightforward version of the 'new method' is redundant; focus Phase 2 only on hard missingness, positivity, and inferential guarantees. **Gate C:** if there is neither an artifact nor a distinguishable under-controlled failure, reconsider the project. Empirical thresholds for 'nontrivial' are exploratory only; never refit thresholds after results.

## Nonidentification / limits

Even if period-specific observed prevalence and every recorded source/page label are known, the prevalence among **unobserved texts** can be arbitrary without assumptions. Therefore no method can identify an unrestricted change in the full historical publication or speaker population from the observed texts alone. Partial-identification bounds are existing statistical methodology, not an invention in this project. Phase 1 can demonstrate the failure mode, not prove a solution.

## Verification and reproducibility

Source ZIP SHA256, extraction manifest, code SHA256, feature list, issue assignments, per-method results, and an independent verify script. Reproduce all 1885 page-level counts from ZIP, check all 400 draws, recompute one full seed and one independent deterministic replay of final CSVs; release no unlicensed raw newspaper texts in the research ZIP. Every post-freeze bug fix must be logged before looking at altered outcome summaries.

Phase-2 advance requires an honest novelty ledger showing prior work on topic/genre confounding, semantic drift uncertainty, partial ID, and synthetic semantic benchmarks. Phase 1 is completed only after QA and decision log. Future phases remain unexecuted.
