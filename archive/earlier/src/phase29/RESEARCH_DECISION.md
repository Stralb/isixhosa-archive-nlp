# Phase 29 research checkpoint (24 October 2026)

## Objective
Investigate whether source-attribution patterns in corrected historical isiXhosa newspapers really require discourse word order, whether they are trivially caused by serial text reuse, and whether the previously observed tokenizer-specialist efficiency is only a word-length composition artifact. Preserve all earlier phases and refuse cultural or sentiment interpretations not supported by expert evidence.

## Outcome
**Phase 29A:** On the previously evaluated issue-held-out 1888 set, newspaper-source prediction from *word-length categories and their short-range arrangements alone* reached 66.53% issue-balanced accuracy (bootstrap 95% CI 59.22–73.43%). Length-bin proportions alone gave only 57.74% (CI 51.38–64.16%), an interesting post-hoc contrast but not a preregistered statistical difference. The original character-within-word model scored 76.40% (95% CI 70.81–82.31%). Shuffling words did not change it at all, **by construction**: `char_wb` features do not cross token boundaries. The original H29a test was invalid as a test of natural language ordering, and was not passed off as a genuine negative research result.

**Phase 29B corrected prospective study:** On the same held-out 1888 issues, an explicitly order-sensitive word-unigram+bigram model scored 73.91%, versus 71.98% for the unigram model. The improvement was only +1.94 pp with paired issue-bootstrap CI [−0.24,+4.49] pp, below the required 5 pp. Shuffling words reduced word bigram accuracy by +1.46 pp (CI [−0.84,+4.08] pp). Cross-word character model order effect was +0.32 pp (CI [−2.35,+3.07] pp). All three precommitted gates failed. This supplies **no compelling evidence** of a source-signature improvement attributable to the tested local word-order features; it does not establish that phrase/discourse differences do not exist.

**Cross-issue repetition exclusion:** Exact normalized 13-token sequences repeated in distinct issue IDs occurred within 23 of 581 windows, including 12 of 207 test windows. All 12 repeated test windows were Imvo; none were Isigidimi. After excluding them, 363 of 374 training windows and 195 of 207 test windows remained (558 windows total); the originally frozen classifier architecture was refitted on those retained training windows (11 train excluded). The repeated-passage-excluded model gave 74.35% held-out issue-balanced accuracy (CI 68.40–80.62%). On the **same retained test windows**, the original full-trained model scored 76.24%; paired difference +1.89 pp (CI −1.08,+5.34 pp). These overlaps do not explain most of the source signal in this sampling frame, but other unobserved repeated paragraphs and genres may matter.

**Phase 29C, linked to tokenizer results:** The 1,242 fixed (207 page × 6 tokenizers) checks proved that SentencePiece piece counts add across individual words in these windows and exactly reproduce saved Phase26 counts. BPE specialist savings averaged +0.9039 pieces per 100 alphabetic characters for Imvo and +0.8902 for Isigidimi (3 tokenizer seeds). Standardizing to the pooled newspaper-equal **character-weighted length-bin mixture** gave +0.8943 and +0.8833 respectively: changes of −1.06% and −0.78%, far smaller than the precommitted 50% mediation gate. Issue/tokenizer-seed bootstrap 95% intervals for standardized effects: Imvo [+0.094,+2.126], Isigidimi [+0.076,+1.424]. Thus gross word-length composition is NOT sufficient to explain the source-specific segmentation efficiency in this sample. This is descriptive reweighting, not causal mediator identification.

## Falsifiable claims and decisions
| Test | Decision | Why |
|---|---|---|
| H29a original word shuffle vs char_wb | **INVALID TEST**, not a finding | Feature function is order invariant by construction |
| H29b length category signature | SUPPORTED (conditional to old data) | 66.5% and lower issue-bootstrap CI >50%, above 60% threshold |
| H29c repeated-text exclusion | SUPPORTED (conditional) | 74.3% and lower CI >50%, above 65% threshold |
| H29B1 bigrams > unigram by >=5 pp | NOT SUPPORTED | Difference 1.94 pp, interval includes zero |
| H29B2 word-bigram word-order loss >=5 pp | NOT SUPPORTED | Difference 1.46 pp, interval includes zero |
| H29B3 cross-word-char order loss >=5 pp | NOT SUPPORTED | Difference 0.32 pp, interval includes zero |
| H29C C1 specialist effect after length-standardization | SUPPORTED (conditional) | Both confidence lower bounds >0 |
| H29C C2 >=50% reduction after length-standardization | NOT SUPPORTED | 0.8–1.1% observed reduction |

## Responsible NLP interpretation and boundary
The corpus was corrected by the IsiXIT project and contains historical orthography. Distinct source patterns can be detected without word identity, but identity/classifier performance **does not** measure political agency, cultural worldview, sentiment or diachronic semantic change. Even small local word-order effects are not a test of historical syntax, because short fixed-window models, n-gram sparsity and article segmentation impose limitations. 1888 was repeatedly studied in earlier phases, so this is not independent confirmation. Note 1886 Isigidimi missing from sources. No experts were recruited, no labels were fabricated.

## Archive metadata gate
Schoots et al. (2023, pp.60-61) document original human-coded 1885 *article-type* and *topic* labels for both newspapers. The provided page-level language-view ZIPs have no such item labels. On 24 October 2026 the current live Omeka S archive welcome page returned a JavaScript bot-verification page, and direct item/API access was not available via our web tool. An alternate public index Emandulo mirrors edition-level converted-text records but not the article-level controlled-vocabulary labels. Therefore Phase29 still **does not provide genre-controlled inference**.

## Publication-oriented next decision
1. Obtain existing 1885 article-level labels from the archive curator, including article IDs, issue date and page/position, topic/type controlled vocabulary, plus transcript span or article text to align them. Verify alignments before any causal/cultural claim.
2. If metadata cannot be acquired, focus the paper on **representational measurement under historical-newspaper domain shift**: corpus → learned BPE vocabulary → source-specific segmentation → neural language-modelling changes, with source-attribution/length results as safeguards against overinterpreting language or cultural change.
3. Do not draft a purported cultural-evolution or sentiment paper without an independently defensible construct or existing expert-coded labels. No paper was drafted in this phase.

## Provenance
- Schoots et al. (2023). *IsiXhosa Intellectual Traditions Digital Archive: Digitizing isiXhosa texts from 1870-1914*. https://aclanthology.org/2023.rail-1.7/
- Prior phases 26-28: sourced text and fixed BPE models supplied with the handoff; original results unchanged.
- `PRECOMMIT.md` and separate pre-result Phase29B/29C amendments; `FROZEN_INPUT_SHA256.txt`; scripts, fixed processed inputs, saved predictions, independent verifier.
