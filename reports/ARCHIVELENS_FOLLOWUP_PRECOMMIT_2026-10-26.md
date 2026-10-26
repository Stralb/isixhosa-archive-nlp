# ArchiveLens follow-up: calibration and structural attribution — frozen before analysis

**Protocol recorded 26 October 2026, after ArchiveLens's original results were known and before these follow-up outputs are calculated.** Status: **post-hoc conditional extension**. This is not independent historical validation. It does not modify the primary tokenizer F1–F6 tests or the original ArchiveLens estimates.

## Motivation and exact scope

The existing ArchiveLens audits 120 within-word four-character patterns selected on 1887, comparing corrected 1885 and 1888 `XhTxt` text views in each of **IMVO** and **ISIG** under character-position mass (`word_mass`), equal page, equal issue and equal month.

Existing warning counts (known before this add-on): IMVO 23/120 point-sign reversals and 9/120 support losses; ISIG 4/120 point-sign reversals and 2/120 support losses; neither newspaper has a case where opposite directions are both supported at the original bootstrap threshold. These known values MUST NOT be marketed as a validated historical-semantic change.

Two new questions:

1. **Calibration:** Are temporal-pattern changes, estimator-sensitive differences and the *number of sign-reversal flags* unusual under a reference null that keeps issue clusters and calendar-month composition intact? What remains after correcting for 120 pattern comparisons?
2. **Attribution:** Which historical publication-issue or page-position exposures and exact repeated passages materially influence the alerts? Does excluding these exact repeated passages alter the flagged temporal directions?

## Fixed inputs, features and representations

- Reuse the original `data/TxtLangFiles1870-1894_TB.zip` (SHA256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`), and exactly the 120-ranked 1887 patterns in `results/archivelens/candidate_1887.csv`. Do not reselect or interpret them as human-verifiable roots.
- Use all eligible (>=40 ASCII `[a-z]+` words) 1885/1888 `XhTxt` pages in the existing instrument. Cluster pages by newspaper and publication issue/day; preserve an issue's pages together.
- Use same four existing point estimands and the same `per 10000 within-word 4-gram positions` units. Report no extra sentiment, language-ID or semantic labels. Original 3,000 bootstrap intervals remain unchanged.

## Experiment 1 — fixed calibration method

- Use **4,999** reproducible random permutations with Python `numpy.random.default_rng(20261027)`, independently within each newspaper and **within each calendar month** across all 1885+1888 issue clusters. Preserve each month/year's number of issues, assign whole issues to pseudo-year bins, and keep their page groups and word counts intact. Repeat the same pseudo-year assignment for all 120 patterns and four estimands to retain dependence structure. **Never shuffle individual pages.**
- `null_1`: two-sided year difference in `word_mass` units per pattern `|Delta_mass|`.
- `null_2`: absolute maximum difference in year change against the three alternative estimands, `max(|Delta_page-Delta_mass|, |Delta_issue-Delta_mass|, |Delta_month-Delta_mass|)`. This is a sensitivity magnitude diagnostic, not proof that one estimator is correct.
- `null_3`: total number of patterns with the existing ArchiveLens point-sign reversal definition (`mass*alternative<0`); compare observed with the full permutation distribution. Report upper and lower empirical tail probabilities, mean and 95% empirical null range. **Do not claim "warnings are better than chance" if they are not elevated.**
- For `null_1` and `null_2`, per-pattern Monte Carlo p-values `(1+#perm(null>=observed))/(4999+1)` (two-sided `null_1` via absolute magnitudes). Adjust separately **within each newspaper and null family** across 120 patterns using **Holm FWER** and **Benjamini–Hochberg (BH) FDR**; report both without implying dependence-free BH guarantees. Criterion `adjusted_p <=0.05`.
- No multiple-testing correction is retroactively applied to existing bootstrap intervals. Their support-loss flags remain **descriptive**, not newly validated false-positive rates; no nested bootstrap for each null replicate is claimed.
- Exchangeability of real historical years is a working permutation reference, **not a credible statement that years really have equal distributions**; editorial and historical change violate a literal sharp null.

## Experiment 2 — fixed attribution method

### Publication-issue influence

For **every** original flag (union of sign reversal or original support loss), compute exact leave-one-publication-issue-out (LOO) year-change estimates for each of the original four estimands. For each issue in either evaluated year, remove all its eligible page views together. If removing the sole issue in a month makes the month-equal estimand undefined, record that estimand as unavailable, do not interpolate; the other three estimands still apply. Report:
- The maximum absolute change in `word_mass` year-effect on any one-issue removal and the identifying issue ID/year.
- Whether any single-issue removal changes the `word_mass` sign; how many issues would do so.
- The same maximum influence for `page_equal`, plus whether its sign flips for any leave-one-issue-out; include n_issue in each source/year.
- An issue is *influential* mathematically, not identified as its author/cause/editorial intent. Top three issues for each flagged pattern ranked by absolute baseline-contrast shift (deterministic issue-ID tie breaks).
- Optionally include a compact issue attribution table for all 240 patterns for comparison, while main figures focus on flagged patterns.

### Exact repeated text

- On the same eligible 1885/1888 `XhTxt` pages, **within each newspaper**, find normalized **13-consecutive-word** sequences appearing in two or more **distinct publication issue IDs** (same-issue repetitions alone do not count). This is a conservative **exact repeated-passage** detector, not an OCR quality score or general boilerplate classifier.
- Mark all word positions covered by any such 13-word repeated sequence in each evaluated page. Remove the marked word occurrences **without joining two remaining fragments** or creating any synthetic cross-word 4-grams. Recount within-word 4-grams and denominators on unmasked words.
- Retain pages with at least 40 unmasked ASCII words and a nonzero within-word char4 denominator; report the **number and proportion of words removed** as well as the page/issue support after removal, separately by newspaper/year.
- Recompute all four temporal point effects on remaining pages and common calendar-month support; compare with original differences and signs for each flagged pattern. Quantify number of original alert patterns whose baseline direction reverses or whose original baseline/alternative sign disagreement disappears. **Do not treat shifts as caused by duplication alone** when eligibility/support changes. No additional claim of statistical support from this sensitivity unless separately bootstrapped.

### Page-position exposure

- Keep the original first-page-exclusion flag as a baseline. Additionally, for each source/year calculate positions 1, 2, 3, 4, 5+ as separate descriptive bins and report page count, word mass, char4 opportunities; compare proportion of the target pattern counts contributed by page 1 and by non-page-1 pages, for flagged features. Not matched article genre: labels are scanned/corrected page positions.
- For any prominently discussed pattern show both year changes and the extent to which the prior page-1 exclusion changed them, without claiming masthead/document genres were identified.

## Artifact/verification requirements

1. Commit this protocol **before running** both follow-up tests.
2. Implement a deterministic CLI `src/archivelens/followup.py` using only existing pinned dependencies and original `audit.py` data structures; don't change primary `audit.py` or primary data/results.
3. Synthetic unit tests: permutation never splits an issue's pages; known year/estimand effects and sign-reversal count; BH/Holm handling; exact duplicate sequence masking across distinct issue IDs but not within one issue; LOO issue influence checks.
4. A second independent implementation or explicit direct recomputation cross-check of at least the complete 120-pattern observed permutation input contrasts, repeat-shingle eligibility/mask and selected issue LOO effects. Report exact maximum error/cross-check method.
5. Outputs under `results/archivelens/followup/`: reproducible provenance and summary JSON, per-pattern calibration CSV, per-flag LOO CSV, duplicate-robustness CSV, page-position table and one paper-ready figure (PNG and vector SVG). Also `reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md` with honest negative results and limitations.
6. Add permanent GitHub CI check that regenerates/compares saved follow-up text artifacts deterministically. For speed use cached computed permutation result only if its exact source and deterministic script can be independently confirmed; avoid saying it was fresh-regenerated if not.
7. Never edit/falsify frozen scientific outputs. Treat existing results as post-hoc conditional, not a separate untouched external sample.

## What counts as a contribution?

A tested instrument with calibrated warning counts, pattern-level uncertainty accounting for multiple comparisons, and traceable archive-structure influences on its warnings. A negative calibration result still improves the honesty of the instrument and should be fully reported. It does **not** establish human cultural change, true semantic shifts, language discrimination, or measured fairness harm.

Prior methods include clustered permutation, multiple testing, leave-one-cluster-out influence, duplication masking and sampling-estimand sensitivity. Distinctive contribution is their reproducible integration in this historical South African isiXhosa archive with source- and year-specific empirical results.
