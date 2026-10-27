# Phase 30: Identical-word controlled analysis of historical-source tokenizer specialization
Frozen: 2026-10-24, before any Phase 30 segmentation frequency association or placebo-tokenizer results were calculated.

## Question
Does newspaper-source-specific BPE efficiency remain when the held-out surface word is literally identical across newspapers, and can training-frequency differences account for which frozen tokenizer segments that exact word more compactly?

## Scientific boundary
Phase 26-29 repeatedly examined 1888. This is a *prospectively specified follow-up on already-inspected data*, not an untouched confirmatory validation of a new cultural or linguistic claim. No true genre labels, morphological analysis or expert annotations are assumed. Same string = same observed orthographic form, not verified same meaning.

## Fixed data
- Existing Phase26 90-token isiXhosa excerpts; 1885/1887 source-specific tokenizer training; 1888 evidence windows, issue-clustered.
- Six previously trained SentencePiece BPE models, vocabulary 1000, three paired seeds per source. The two source-specific training text files for each seed are frozen (60,000 alphabetic characters each). No retraining of experimental source-specific models.
- Token matching uses exact lowercased Phase26 surface forms; punctuation preserved only as existing tokens. No stem labels.
- Define 'shared held-out form' = exact string appearing in both 1888 newspaper subsets. H30A further requires total count >=2 across the corresponding two *training files* (eligible counts ~700 forms/seed from feasibility audit). Test inclusion fixed before scores.
- Separate 'exclusive held-out form' = occurs only in one 1888 subset. These are not necessarily source-exclusive across all years.

## Hypotheses and statistics
H30A frequency alignment: For each seed and eligible shared form w, x=log((n_IMVO,train(w)+0.5)/(n_ISIG,train(w)+0.5)), y=pieces_ISIG(w)-pieces_IMVO(w). Compare x and y using Spearman rho over distinct *types*, NOT frequency-weighted observations. Predicted positive. Required: rho >= .10 in all three seeds, with length-and-pooled-training-frequency-stratified permutation p <= .05 in all seeds (2000 permutations per seed; RNG 30024+seed). Group length bins [1-3,4-6,7-9,10-12,13+] and pooled frequency bins [2-3,4-7,8-15,16+]; permute x within cells where size >=2; retain x otherwise, account for singleton invariance. Also report seed results unthresholded and central 95% type-bootstrap rho intervals (1000 resamples).
H30B identical-word residual: On the subset of 1888 *tokens* whose exact form occurs in both held-out newspaper sets, source-specific BPE specialist advantage (other-model pieces - own-model pieces) per 100 alphabetic characters is positive for *both* newspapers. Primary decision: issue+seed bootstrap lower 95% CI >0 for both (2000 draws; RNG 30324). Report full-sample advantage and proportion of signed aggregate advantage contributed by shared tokens; note ratios unstable if total effect near 0. This is not a human-fairness judgment.
H30C randomized-source placebo: Construct pooled training tokens separately within each seed from the frozen two newspaper training texts. Deterministically shuffle with RNG 30124+seed and partition tokens into two *source-blind* 60,000-alphabetic-character streams by greedily filling to exact character budget, trimming only at a deterministic last token if necessary; if arbitrary substring truncation would be required, instead stop at largest whole-token budget <=60000 and use identical budgets across both streams. Fit matched SentencePiece BPE 1000-vocab placebo-A and placebo-B (identity normalization, character_coverage 1, same other settings). Compute cross-newspaper symmetric specialized advantage S = own-source savings Imvo + own-source savings Isigidimi on all 1888 evidence tokens. Placebo orientation A=IMVO, B=ISIG fixed before results. Decision gate: observed S exceeds twice the mean of absolute placebo S across three seeds, and a paired issue+seed bootstrap of (observed S - placebo S) has lower 95% bound above 0. Note random partition does NOT perfectly equate every type frequency; it is a baseline for chance allocation, not a definitive causal mediation analysis.

## Checks and negative controls
- Verify exact token-additivity against every frozen Phase26 page/seed/model (1242 checks), abort on mismatch.
- Report shared-type count, token coverage per source and sample months/issues; do not cherry-pick words or results.
- Report H30A again excluding exact paper titles and author/editor name substrings specified before scoring: {imvo, isigidimi, xosa, jabavu, bokwe, gqoba, makiwane, hofmeyr}. This is sensitivity only; the strings may have genuine historical significance.
- Produce full per-type and per-issue results, hashes, reproducible scripts and independent validator; no paper writing.

## Decision
If H30A/B/C hold, strengthen the operational explanation 'source composition -> differential frequency exposure -> exact-form token segmentation'; do not claim that word frequency alone *causes* each merge. If fails, reject the corresponding explanation and preserve the original controlled tokenizer/domain result as narrower.
