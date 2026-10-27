# Phase 29C - word-length composition and tokenizer-specialist advantages

Saved before any Phase29C score computation, following Phase29A's positive but semantically uninformative length-bin classifier result (itself previously inspected). This is an *exploratory prospective follow-up*, not independent data confirmation.

## Question
Is the Phase26 1888 specialist-BPE advantage mostly explained by differing *surface-token length distributions*, or does source-specific efficiency remain after standardizing the length-bin composition?

## Input and models
Use exactly the 207 unchanged 90-token 1888 windows from Phase26; BPE_IMVO_seed{0,1,2}.model and BPE_ISIG_seed{0,1,2}.model (fixed 1000 vocabulary, training 1885/1887). Measure pieces for each token independently; verify additive piece counts against previously saved whole-window SentencePiece output exactly. Abort if any mismatch. No new tokenizer training.

## Estimands
For each newspaper s and seed q, specialist-vs-other advantage in pieces per 100 alphabetic characters: for Imvo `(P_Isig - P_Imvo)*100/C`, and reverse for Isigidimi. Group tokens into exact bins of length 1-3, 4-6, 7-9, 10-12 and 13+. Weight bins by alphabetic-character mass. The counterfactual standardized advantage uses the **common character-mass weights**, defined by the equal-weighted mean of newspaper-specific character shares on the complete 1888 sample, multiplied by each source's measured within-bin specialist advantage. This is a descriptive standardization, not a causal estimate of genre or language evolution.

## Uncertainty
Issue-cluster bootstrap within each newspaper (2,000 resamples) and draw one of the three tokenizer seeds in each resample. RNG=29324. Keep common length weights fixed to avoid changing the target standardization across resamples. Report raw and standardized estimates, bin share, within-bin savings, and CI conditional on available issues and tokenizer seeds.

## Decision gates
C1 (residual specialist effect): length-standardized advantage positive for both newspapers, with both bootstrap 95% intervals lower >0.
C2 (strong length mediation): standardized advantage for BOTH sources decreases by >=50% relative to their raw character-weighted advantage. Precommitted but not assumed.
If more than 1% of windows fail token-additivity, treat all numerical length results as inconclusive and debug without claiming success; even one difference must be reported as a failure of exact additivity claim.

## Claim boundaries
The newspaper samples differ in author/genre/topic/editing and no human article-type metadata is available. Token length is not a genre or language annotation. The test is repeated on previously used 1888 text, and standardization cannot establish a historical meaning change.
