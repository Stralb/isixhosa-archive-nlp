# ArchiveLens follow-up: reproducible calibration and attribution

**Experimental status:** completed observational, **post-hoc/conditional** analysis of historical years previously inspected in other phases. Do not call the new 1885/1888 trends untouched external replication, independent linguistic truth or measured societal fairness.

## Question 1 — Do the sign reversal flags exceed a plausible null?

The new `followup.py` program performs **4,999** within-calendar-month permutations of entire publication-issue clusters. Each year's issue count within each month is preserved. It recomputes both (a) the absolute historical temporal difference in character-mass rate, and (b) the **maximum change in the year contrast across four sampling estimands**.

It calibrates **two distinct questions** and avoids conflating them:

- **Count benchmark:** Is the total number of point-sign reversal flags (out of 120) unusually high under an exchangeable-year-label reference? This is not a measured individual-warning false positive rate.
- **Pattern tests:** Which year effects or estimator gaps exceed their own permutation distribution **after Holm familywise and BH FDR correction** within a predeclared family of 120 patterns? This does not retroactively adjust the original 3,000-bootstrap intervals or prove historical semantic change.

The exact 1887-selected patterns and pre-existing 1885/1888 text files are unchanged.

## Question 2 — Which archive structures affect warnings?

For each originally flagged pattern (union of sign reversal and bootstrap support loss), compute:

- **Leave-one-publication-issue-out** change in character-mass and equal-page temporal contrasts. Keep all pages of an issue together. Report the three issue IDs with strongest arithmetic influence, and whether any removal flips direction.
- **Cross-issue exact repeated-text masking:** Find normalized 13-consecutive-word shingles repeated in different issues **within the same newspaper**, remove all covered words without joining separated words, re-evaluate eligible pages and four sampling estimands. Coverage and sample attrition must be reported.
- **Page positions:** Summarise page 1–4 and page 5+ opportunities, and for originally flagged patterns the fraction of pattern occurrences in each group. Page position is NOT annotated article type or expert editorial context.

This provides **traceable sensitivity witnesses**, not causal identification of an author's editorial aims.

## Code and reproducible commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s src -p 'test_*.py' -v
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .
python src/archivelens/verify_followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify
```

The independent `verify_followup.py` **does not import either followup.py or audit.py**. It independently re-reads all 594 evaluation pages; replays 4,999 permutations per newspaper with separate aggregation code and multiplicity correction; verifies all 114 top issue influence calculations; and checks repeated-word masking on every page using exact **tuples of 13 words**, separately from the main BLAKE2b-digest implementation. Synthetic tests cover impossible within-issue splits, controlled nulls, multiple testing, issue influence and exact repeated words.

## Results, formats and limitations

- [Full findings](../../reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md) and [frozen follow-up protocol](../../reports/ARCHIVELENS_FOLLOWUP_PRECOMMIT_2026-10-26.md)
- `results/archivelens/followup/summary.json` machine-readable summary and protocol details
- `calibration.csv` (240 feature/source p-values, Holm/BH adjustments and observed gaps)
- `attribution.csv` and `influential_issues.csv` (flag-level influence and top 3 issue witnesses)
- `duplicate_coverage.csv`, `duplicate_sensitivity.csv`, `page_duplicate_masks.csv`, `page_positions.csv`
- `results/figures/archivelens_followup_calibration_attribution.png` (also SVG)

The headline **negative result** is crucial: original point-sign reversal warnings do **not** exceed the permutation benchmark in either newspaper. Significant estimator-gap patterns are a separate test of **magnitude sensitivity**, not confirmation of the original warning count. Neither analytic family establishes actual linguistic roots, meaning changes, cultural harm or social discrimination.

Source/citation considerations: corrected digital `XhTxt` view labels are unadjudicated; no independent genre annotation or raw OCR/confidence is used; the dataset has missing/historically uneven preservation; the title/year composition and genre remain confounded; strong dependencies exist among overlapping four-grams; Holm/BH are applied separately for each family, not a joint correction across all multiple families.
