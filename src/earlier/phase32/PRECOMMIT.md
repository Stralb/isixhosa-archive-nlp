# Phase 32 prospective protocol — language visibility and sampling estimands
Date: 2026-10-24. Saved before extracting any phase-32 outcome statistics.

## Research question
In the supplied 1870–1894 historical newspaper archive, how do (a) publication and period, and (b) the unit used to sample historical text, change the amount of archive-assigned English-view material seen by an NLP system? This is a Responsible NLP data-representation study, NOT a claim about historical linguistic attitudes or speaker identity.

## Data / labels / controls
- Only supplied `TxtLangFiles1870-1894_TB.zip` and its original tracker plus corresponding `XhTxt` and `EnTxt`; filename-based paper/date/page parsing.
- Source-provided `XhTxt`/`EnTxt` are **block-level assignments**, not verified word-level language labels. Call results 'English-view word mass' rather than true language identification. Do not use unverified labels from OCR or language-model predictions.
- Count whitespace-delimited words in each original field and compare against tracker. Raw page counts used as integrity diagnostics; primary denominator is Xh+En classified words.
- Issues are publication-date groups; pages in one issue are correlated. The archive's missing Isigidimi year 1886 will never be interpolated.
- Predefine overlapping publication comparison as calendar year 1885, 1887, 1888; shared year-month cells only. No causal publisher inference.
- Imvo period comparison: 1885–1888 vs 1889–1894, excluding startup year 1884. Evaluate page coverage and heterogeneity separately; later Isigidimi comparator does not exist.

## Outcomes and preregistered hypotheses
1. **H32a / source visibility:** Mean per-issue archive-English share in Imvo minus Isigidimi, averaged equally across common year-month cells, is at least +10 percentage points. Use matched-calendar-cell issue bootstrap (B=3000; seed=3201). The 95% CI must exclude zero; report yearly cells and unequal coverage.
2. **H32b / sampling estimand:** On Imvo during 1885/1887/1888, the word-mass English share minus uniformly page-averaged English share exceeds +5 percentage points; issue-cluster bootstrap B=3000 (seed=3202), with 95% CI above zero. This is not a new statistical phenomenon (Phase 2 explored sampling-unit differences), but a temporal-control and issue-uncertainty check.
3. **H32c / temporal change:** Compare Imvo mean per-issue archive-English share in 1885–1888 versus 1889–1894. Predefine substantial change as abs(delta)>=5 points and issue-bootstrap CI excluding zero (B=3000, seed=3203); direction NOT chosen. Report uncertainty when resampling years (n=4 vs 6) as sensitivity; no causal interpretation.
4. **H32d / text concentration exploratory diagnostic:** In all Imvo years, percentage of total English-view word mass occurring in top decile of pages by English-view count; repeat for Isigidimi. This is descriptive, not hypothesis testing; ranking by English count is defined before observing outcome.

## Planned negative/integrity controls
- Confirm tracker-reported Xh/En counts against current ZIP file contents; examine missing/unclassified words separately.
- Recompute H32b within early and late Imvo separately; check page 1/2/3/4+ decomposition and issue stratification as pre-specified diagnostics.
- Compute year-balanced version of H32c (equal year weight) in addition to issue-weighted contrast so publication frequency does not quietly define historical time.
- No post-outcome tuning of thresholds. A failed hypothesis remains failed.

## Decision gate
If source- or sampling-specific language distributions are stable beyond issue variation, frame as **archive-data curation / visibility**, not opinions. If not, reject broad newspaper or temporal claims. Do not claim a novel historical NLP method or sentiment trends. This stage is distinct from Phase 31 token budget effects.

## Reproducibility
Archive input SHA256 recorded in hashes; saved page and issue aggregates; bootstrap random seeds; exact scripts and validation; exclude verbatim full newspaper text from final portable deliverable.
