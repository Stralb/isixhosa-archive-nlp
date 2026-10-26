# Exploratory follow-up: segmentation/LM coupling and excerpt-length heterogeneity (26 October 2026)

**Status: POST-HOC EXPLORATORY ANALYSIS.** This is an additional interpretation of existing, frozen held-out evaluations; **no new model training, hyperparameter selection, newspaper source, or change to the prespecified F1-F6 decision gates**. The source newspaper is *Imvo Zabantsundu*. The same 1888 and 1889-94 text was examined during earlier research; neither period is now an independently untouched corpus.

## Motivation and scientific question

The final factorial study found opposing segmentation effects in the archive's English and isiXhosa text views, and a narrower, policy-dependent transfer to count-bigram language modelling. The new question is:

> **Do changes in tokenisation compactness consistently track changes in downstream bigram loss for the *same* held-out newspaper texts, and are the representation effects broadly distributed or dominated by particular publication issues and excerpt lengths?**

These are **measurement** questions, not direct measurements of cultural semantics, fairness harm, modern speakers, or neural language models.

## Design and source verification

- Inputs: the original **99,720** Phase 3 held-out segmentation scores and the matched **99,720** Phase 5 count-bigram scores, along with the original final contrast tables, all in `results/raw/` and `results/`.
- An independent computation joins each record exactly on `(family, arm, seed, filename, view, issue, year, period)`, checks matching lengths, text fingerprints, token counts and zero unknown pieces, then forms **24,930** five-seed/family/excerpt sampling-arm quartets.
- For each held-out excerpt calculate the allocation effect from all four arms, `0.5[(WP-PP)+(WW-PW)]`. The segmentation measure is pieces per **100 alphabetic characters**; the count-bigram measure is **bits per alphabetic character**. Both have lower-is-better direction.
- **Issue-level analysis:** first aggregate paired records within each publication issue by character length, then average five fixed tokenizer seeds before computing correlations or direction counts. Pearson and Spearman associations are descriptive. For Pearson's uncertainty, resample **publication issues** (5,000 replicates); DO NOT count pages or seeds as independent newspaper issues.
- **Length analysis:** group evaluated **excerpt lengths**, not entire newspaper-page lengths, into four source-character quartiles defined separately by period and archive view, with the same boundaries for BPE and Unigram. Calculate source-character-weighted effects and **paired issue-bootstrap** CIs within each stratum. These are post-hoc sensitivity checks, not new confirmations.
- Every reported full-sample allocation, within-view and whole-policy effect was recomputed from the paired raw rows: **maximum absolute difference from the original published point estimates = 2.33e-15**. The original results and experiments were not changed.

## A. Do segmentation changes predict bigram effects?

All estimates below are **1888 publication-issue-level**; r uses the 52 issue-level, five-seed mean factorial allocation effects, not individual model/page pseudo-replicates.

| Tokenizer/view | Pearson r [95% issue bootstrap] | Issues where segmentation and bigram signs differ |
| --- | ---: | ---: |
| BPE / English view | +0.687 [+0.528, +0.809] | **1 / 52** |
| BPE / isiXhosa view | +0.470 [+0.306, +0.638] | **29 / 52** |
| Unigram / English view | +0.795 [+0.641, +0.895] | **2 / 52** |
| Unigram / isiXhosa view | +0.639 [+0.446, +0.787] | **5 / 52** |

**Key distinction:** On BPE isiXhosa-view excerpts, segmentation became less compact in **50/52 issues**, while count-bigram loss improved in **31/52**. In **29/52**, they moved in opposite directions *on the same issue*. This is a direct counterexample to treating intrinsic compactness as an adequate downstream proxy, **not** a blanket claim that these metrics are unrelated (Pearson r is still positive). BPE English-view effects both improved in 51/52 issues.

The pattern of direction discordance persists on the previously inspected 1889-94 slice: BPE isiXhosa has **262/306** discordant issues; BPE English has **9/306**. These are dependent, same-newspaper descriptive observations and should not be called cross-corpus generalisation. The overall 1888 isiXhosa-view BPE bigram allocation estimate is small (**-0.002746 BPC**), and only **3/5 trained seeds** share its negative direction; do not exaggerate it.

## B. Is the phenomenon limited to unusual excerpts?

On **1888** data, the character-weighted BPE English allocation effect remains negative across all four source-excerpt-length quartiles:

| English-view excerpt length | N page views | Segmentation (pieces/100 chars) | Bigram BPC |
| --- | ---: | ---: | ---: |
| Q1, 166-326 characters | 34 | **-1.787** | **-0.0757** |
| Q2, 345-566 characters | 33 | **-1.815** | **-0.0387** |
| Q3, 567-593 characters | 34 | **-2.004** | **-0.0357** |
| Q4, 595-652 characters | 33 | **-1.844** | **-0.0212** |

For isiXhosa-view BPE, segmentation cost increases in **all four quartiles** (approximately **+0.42 to +0.51** pieces/100 characters); its tiny BPC effects vary in sign or have intervals overlapping zero. With Unigram, English segmentation improves in all four quartiles and isiXhosa segmentation becomes less compact in all four.

This supports **breadth of the measured segmentation effect**, not that excerpt length *causes* the impact. The excerpt-length quartiles are post hoc and partly reflect the original first-128-word scoring cap and underlying word lengths. They are neither historical genres nor document-population strata.

The issue-level leave-one-out diagnostic also shows that **no single 1888 publication issue reverses the full BPE English allocation result**: segmentation range **[-1.89560, -1.86224]** pieces/100 chars and bigram range **[-0.038203, -0.036306]** BPC. For BPE isiXhosa segmentation, 1888 leave-one-issue range **[+0.4470,+0.4621]**.

## Scientific contribution and disciplined scope

The experiment deepens the main study's interpretation: **corpus allocation changes representation across archive language views, but the same numerical compactness gain is not a sufficient indicator of downstream modelling efficiency**. A policy can trade improvements across views and also across metrics. The presence of a positive across-issue association does not remove issue-level counterexamples, especially for isiXhosa-view BPE.

**Limitations:** one newspaper; language views unadjudicated; previously inspected test years; basic count-bigram downstream model only; post-hoc new analyses with no multiple-testing adjustment; seed sharing and issue resampling do not quantify all archive/population uncertainty; and some exact 13-word text overlaps with training years. These do not establish effects on cultural beliefs, sentiment, neural model accuracy, or societal harm.

The original two-page report should lead with the prespecified bilingual allocation trade-off and factor decomposition. Use this new analysis only as concise explanatory support in the Discussion, not to replace the primary findings or inflate claims of independent replication.

## Reproducibility

The companion script `src/analysis/metric_relationships.py` recomputes all above measurements and issues the CSV tables/PNG diagrams in `results/exploratory/` and `results/figures/` from **saved raw scores only**. The script also checks all full-sample effects against the original published CSVs and has a `--verify` mode to compare regenerated output evidence. Run:

```bash
python src/analysis/metric_relationships.py --root .
python src/analysis/metric_relationships.py --root . --verify
```

The existing source data and trained models remain unchanged. The original dataset ZIPs are already in the repo `data/` but this follow-up does not re-read or retrain on them.
