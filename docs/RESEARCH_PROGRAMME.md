# Research programme and evidence boundaries

## Study A — controlled training-data sampling

**Question:** In *Imvo Zabantsundu*, can between-view allocation and within-view page selection independently change tokenizer segmentation and smoothed count-bigram loss?

The main 2×2 factorial comparison uses PP, PW, WP, WW sampling policies, BPE and Unigram tokenizers, five seeds and matched train/validation/test text. Training uses 1885 and January–September 1887; smoothing selection uses October–December 1887; 1888 is the primary evaluation. The later years were studied in earlier research and are a dependent sensitivity check.

The central result is a *trade-off*: word-like allocation improves English-assigned-view BPE compactness by approximately 1.88 pieces/100 letters, but increases XH-assigned-view cost by 0.45. An isolated English-view count-bigram benefit is narrower and does not imply stronger neural-model performance.

Evidence: [methods](../reports/methods.md), [findings](../reports/findings.md), [segmentation](../results/segmentation.csv), [bigram](../results/bigram.csv).

## Study B — ArchiveLens measurement sensitivity

**Question:** On both *Imvo* and *Isigidimi*, do 1885–1888 changes in selected surface four-grams depend on whether text positions, pages, issues or months receive equal weight?

The auditor fixes 120 patterns from 1887, compares four estimands and tests whether observed warnings are unusual under 4,999 whole-issue, month-stratified year-label permutations. The number of sign reversals is **not** elevated under this conditional reference; 29 Imvo patterns have Holm-adjusted *magnitude* gaps, while Isigidimi has none under the same procedure. Issue influence, page position and exact repeats are sensitivity analyses, not established historical causes.

Evidence: [ArchiveLens results](../reports/ARCHIVELENS_RESULTS_2026-10-26.md), [calibration](../reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md), [auditor code](../src/archivelens/README.md).

## Relationship and interpretation

Both studies ask what happens when people treat a surviving digitised archive as though its sampling units were interchangeable. **They are conceptually linked, not a proven causal pipeline.** ArchiveLens does not take tokenizer outputs as its input or establish a historical social-fairness mechanism. The input texts are heterogeneous newspaper material with archive-assigned text views.

No claim about historical author beliefs, language identity, general NLP fairness or cultural change is established without independent annotation. The earlier exploratory programmes are indexed in [research history](../reports/research_history.md) and are not automatically additional confirmatory tests.
