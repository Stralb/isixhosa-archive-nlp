# Phase 27 - Research decision (2026-10-24)

## Objective
Determine whether historical isiXhosa period differences generalize across the two newspaper domains and whether manipulating the source composition of tokenizers changes a period-classification inference. The study responds to Phase 26's source signatures and prior BPE/domain-level disparities. It cannot infer historical semantics or political opinion from unlabeled text.

## Protocol integrity
- A frozen prospective primary protocol defined H27a-c before the 1885/1888 period-model scoring.
- A pre-scoring feasibility amendment reduced tokenizer text budget to 40,000 alphabetic characters because the 1887 samples could not support 60,000/source. All nine BPE models were trained exclusively on 1887; no classifier model was tuned on its held-out months.
- 390 1885/1888 90-word excerpts: four month-blocked folds, two training titles, two text conditions, 11 representation variants (word, char, and 3x3 seeded BPE models). 176 trained classifier fits and 17,160 predictions, all saved.
- Issue-balanced period accuracy and a paired 12-month bootstrap with 2,000 replicates.
- This is a prospective analysis of an archive whose 1888 material was repeatedly examined in earlier phases, NOT new fully independent confirmation.

## Primary outcomes

| Hypothesis | Frozen threshold | Observation | Decision |
|--|--|--|--|
| H27a | Both within-newspaper char BA > 60% | IMVO 64.80%, ISIG 61.48% | SUPPORTED as numerically specified |
| H27b | Cross-newspaper char transfer gap >=10 pp, month-bootstrap lower >0 | Within 63.14%, cross 56.97%, gap 6.17 pp, CI [0.93,11.72] pp | NOT SUPPORTED (practical magnitude fails) |
| H27c | At least one cross-newspaper BPE mixture endpoint shift >=5 pp and CI excludes zero | IMVO->ISIG: -1.89 pp, CI [-8.37,+4.32]; ISIG->IMVO: +3.60 pp, CI [-0.25,+7.42] | NOT SUPPORTED |

Original non-masked cross-source issue BA: word 53.50%, char 56.97%, BPE from 0/50/100% Imvo: 50.13% / 50.86% / 50.98%. All model families detect some within-source period contrast, but BPE unigram features do not generalize robustly across newspaper titles in this simple fixed logistic-classifier architecture. Not a statement about transformer models or semantic embeddings.

[PREPLANNED ROBUSTNESS] Removing source/title/political-name cues leaves char within 62.58% and cross 57.33%; BPE cross remains 50-52%. This does not remove genre/topic/orthography confounding.

[POST-HOC COMPARISON] Char vs balanced BPE unigram cross-source gap = +6.11 pp; paired month bootstrap [2.50,10.24] pp. When masked, +6.50 [2.43,10.99] pp. This family comparison was specified only after viewing the primary results and is exploratory.

[DESCRIPTIVE HISTORICAL ANCHOR] Schoots (2023, RAIL 2023, 2023.rail-1.8) previously studied roots such as -Ntsundu and -Mnyama. Our literal substring and exact surface word counts on 90-word windows show large disparities in coverage and sparse Mnyama observations; they provide NO validated meaning labels and do not show a new political-identity discovery. In Imvo, the `ntsundu` substring declines 86.7 -> 48.8 occurrences per 10k tokens between sampled 1885/1888 windows, while exact `abantsundu` declines 23.3 -> 11.7. Same directional trend, no reversal. This is a diagnostics-only measurement caution, not replication of Schoots's referent-restricted root analysis.

## Prospective chronological stress test motivated by first results

After inspecting Phase27's original BPE costs, we noticed a suggestive Imvo difference between 1885 and 1888. We logged **FUTURE_YEAR_PRECOMMIT.md before extracting/scoring the 1889-1894 text**, to prevent promoting that pattern without a new chronological check.

- Six later Imvo years, 1889-1894, with 740 unique windows across 305 issues (34 duplicated window occurrences removed).
- Fixed same 1887-trained seed-matched Imvo-only and Isigidimi-only 1k BPE models; no new training or tuning.
- Issue-level per-100-character specialist advantage averaged equally across years: earlier 1889-91 = +1.305; later 1892-94 = +1.157.
- H27d (>= +0.30 increase and positive issue-bootstrap interval): observed -0.148; stratified issue-bootstrap 95% CI [-0.352,+0.057]. **NOT SUPPORTED**.
- Cue-masked sensitivity: -0.144, interval [-0.366,+0.075], also no increase.
- Important positive observation: relative Imvo-trained tokenizer compactness persists across all six years (annual advantage >0 with all three seeds), but it does NOT continue increasing. Genre and topical composition are still possible explanations.
- The hypothesis was conceived after earlier results; the later years form a new temporal slice of the SAME historical collection, not independent external validation.

## Conclusion

1. Observable newspaper-specific temporal surface-form signals exist, but are modest and only partially transport across publication domains.
2. A tokenizer source-composition intervention changes representation length on the same text. However its effect on this particular year-classification task is not large/reliable enough to demonstrate changed historical conclusions.
3. Post-hoc, character n-grams are more transportable than BPE unigrams on this task, and the result survives source-cue masking. This is not a test of pretrained contextual multilingual models.
4. Source-specific BPE advantage persists into later Imvo years, but the preregistered *growth* hypothesis fails. This rejects a tempting narrative of accelerating representational divergence.
5. We cannot infer changes in culture, ideologies or sentiment. Without article-type/genre control, differences can reflect text composition; the labels are publication/year metadata, not historical attitude labels.

## Publication verdict
This is a stronger *negative-control and measurement-validity* evidence package, not a newly demonstrated cultural-semantic discovery. It supports a possible paper about data composition, algorithmic representation and the limits of computational claims concerning historical isiXhosa. It does not yet meet a top research venue's novelty/interpretability bar.

## Next highest-value gate
Obtain/reuse human-coded **1885 article-type/subtype metadata** described by Schoots et al. (2023, 2023.rail-1.7), align it at item level, and test newspaper-domain differences on matched **genre/topic** subsets. If alignment or later-year topic information is unavailable, limit claims to surface textual distributions; do not replace that gap with fabricated sentiment or translations. Longer-term, verify independent source OCR and historical interpretability with qualified experts if available. No new report/abstract requested.

## Verification
- Primary frozen dataset SHA256: 518868f66de6b95947c84ddd42eb180398dcea0fbb65ee8e3948b3cbae6f689f.
- All Phase27 predictions and tokenizer hashes matched, and four classifier subsets independently refit to machine-precision score equality. Independent pandas issue-level BA aggregation differed by <3e-16.
- Later-year all 4,440 BPE score records re-encoded independently; issue aggregations and all bootstrap endpoints reproduced exactly. Both verifier JSON files report pass.
- 16 unknown subword tokens across nine Phase27 model/source inference checks, 80 unknown subword tokens across all 4,440 later-year scored configurations. Not a zero-OOV claim.

## Core scholarly prior work
Schoots, Jonathan. 2023. 'Analyzing political formation through historical isiXhosa text analysis: Using frequency analysis to examine emerging African Nationalism in South Africa.' RAIL 2023, pp.65-75. https://aclanthology.org/2023.rail-1.8/
Schoots, Jonathan et al. 2023. 'IsiXhosa Intellectual Traditions Digital Archive: Digitizing isiXhosa texts from 1870-1914.' RAIL 2023, pp.54-64. https://aclanthology.org/2023.rail-1.7/
