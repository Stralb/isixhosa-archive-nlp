# Phase 28 - research decision (24 October 2026)

## Gate outcome: computational checks pass, but the genre-control gate is still UNMET

**Data audited:** The source-corrected isiXhosa paper excerpts are 581 unique 90-word windows from 1885, 1887, 1888, produced and inspected during Phase 26. Historical year 1886 is missing for Isigidimi and was not imputed. No expert language annotations were added. Schoots et al. (2023, RAIL, pp.60-61) describe scholar-coded item types/topics for both newspaper collections in 1885, but the labels are not in the downloaded TXT files; the Ibali site required a JavaScript browser challenge during this run, so we could not reliably fetch or align the labels. The current test uses a **lexical similarity proxy**, explicitly NOT historical topic or article genre ground truth.

## New question
Do previously measured newspaper-source differences persist if excerpts are matched within the same calendar month on surface vocabulary, and do source-specific BPE tokenizers still offer compactness savings on those matched texts?

## Design and integrity
- `PRECOMMIT.md` and three Phase26 inputs were hashed before the new matching and scores.
- TF-IDF word-unigram vocabulary learned only on 1885+1887 text, with predeclared publication/person cue removal, `max_df=0.35`, `min_df=2`, 3,129 learned word features.
- Hungarian optimal one-to-one cross-newspaper pairs within each year/month: 80 pairs (1885), 85 (1887), 84 (1888), 249 pairs total. The 1888 paired set has 168 pages across 49 Imvo and 12 Isigidimi newspaper issues.
- 500 same-month random one-to-one controls measure the improvement in lexical matching and chance selection variability.
- Reused exact already-frozen Phase26 classifier outputs and BPE costs on held-out 1888. No retraining or tuning against Phase28 outcomes. Therefore **no fresh independent generalization claim**.
- Twelve-month cluster bootstrap (2,000 resamples) for estimated held-out accuracy and BPE matching-source advantage. The confidence ranges are conditional on this collection and design.
- `verify_phase28.py` independently reconstructs every matching cost, proves global pair optimality in 36 year-month cells, and recalculates issue-level model scores and BPE savings; all gates passed with zero numerical discrepancy.

## Results and predetermined gate classifications

| Gate | Predetermined criterion | Observed | Classification |
|---|---|---|---|
| H28a matching feasibility | gain >=0.025 cosine and positive month bootstrap interval | 1888 matched cosine .08894 vs random .03552; gain +.05342, CI [.04417,.06901] | PASS **as proxy feasibility only** |
| H28b source predictability | char model issue-balanced accuracy >.65 and CI lower >.5 | matched .7500, month bootstrap [.71155,.79117]; full .76397; 500 random-matched mean .76298 | PASS |
| H28c BPE specialization | source-specific saving positive for both and CIs exclude zero | Imvo +.6598 [.3846,.9448], Isigidimi +.9107 [.7093,1.1123] pieces/100 alphabetic chars | PASS |

Word-unigram model: full issue-balanced 71.98%, matched 70.03%. Random month-matched mean 71.69%.

For Imvo, BPE specialist benefit falls from +.886 to +.660 under matching. For Isigidimi, the matched and unpaired benefit is unchanged at +.911 because **all 84 eligible Isigidimi test pages are selected in the rectangular matching; Imvo is the larger source**. Never interpret unchanged Isigidimi as independent proof of invariance.

**Publication-cue masking sensitivity:** Previously inspected masking specifications were reapplied with the same Phase26 settings (this is NOT another preregistered hypothesis test). Full newspaper char classifier 75.62%, matched 72.85% issue-balanced accuracy. The drop is meaningful descriptively but does not reveal political meaning or article category.

## Critical weaknesses and falsification boundaries

1. Matched-pair absolute cosine remains **very low**: mean .0889, median .0726; only 25% of 1888 matches have cosine >= .1, and only 7.1% >= .15. The optimization necessarily outperforms random matching; this is not evidence the resulting pairs truly share genres.
2. Genre/type, topic and editorial vs letter vs advertisement labels have NOT been controlled. The published scholar-coded 1885 metadata must be retrieved and aligned at the item level to test that.
3. Reusing the Phase26 1888 test makes this a prospective *sensitivity analysis on previously studied data*, not an independent validation sample. Numbers should not be promoted into peer-review-ready novelty claims.
4. Classification is of **newspaper source**, not author intention, cultural worldview, sentiment, ideology or political agency. Distinguishable newspaper surface distributions cannot support causal claims about the establishment of Imvo or nineteenth-century cultural change.
5. Masking can remove real political identity terms, which can attenuate the actual target phenomenon, not simply confounds.
6. Imvo has more windows/issues than Isigidimi; even after equal pair counts, issue balance and monthly clustering are necessary. Resampling only 12 months is imprecise for broad population claims.

## Decision
The replicated source signatures and tokenizer source advantages remain observable after coarse lexical matching. This is **modest triangulation of earlier representation-bias evidence, not yet a novel cultural NLP result**. Do not claim genuine genre control; do not draft the paper. The best next step is to secure existing hand-coded 1885 article-level metadata and link each article record to the corrected page/paragraph, with explicit alignment quality audit. If those item labels are unobtainable, use the measured limitation to refocus on corpus representation and measurement validity instead of claiming cultural change.

## Scholarship
- Schoots, Ngwendu, de Wet & Muftic (2023). IsiXhosa Intellectual Traditions Digital Archive: Digitizing isiXhosa texts from 1870-1914. RAIL, pp.54-64. https://aclanthology.org/2023.rail-1.7/
- Schoots (2023). Analyzing political formation through historical isiXhosa text analysis. RAIL, pp.65-75. https://aclanthology.org/2023.rail-1.8/ . Earlier frequency comparisons already study political roots; our experiment is a measurement/control diagnostic, not a novel political-root analysis.
