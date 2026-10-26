# Research decision — Phase 2 closed

**[RESULT]** All 40 tokenizer fits completed successfully (2 algorithms x 4 factorial arms x 5 seeds), with 500k +/-30 alphabetic characters each and learned vocabulary size exactly 2,000. Each algorithm reused the *same sampled training input* for a given arm/seed. No 1888-1894 text was read by the new Phase 2 experiment.

**[DESIGN DECISION]** The sampling factors and hypotheses F1-F6 remain unchanged from the Phase 1 SHA256-frozen protocol. Phase 2 has not produced the F1-F6 test outcomes; these belong to later phases.

**[RESULT]** On 1887 Oct-Dec validation windows that share **no exact 13-word sequence** with the matched training view, an identical smoothed bigram LM, with the same 545 raw training excerpts for every fit, was assessed at alpha 1/10/100. All scores were averaged equally across archive views, tokenizer algorithms, sampler arms and model seeds.

| alpha | mean duplicate-safe validation bits/character |
|---:|---:|
| 1 | 3.124603 |
| 10 | 2.827858 |
| **100** | **2.732927** |

**[DECISION FROZEN]** Select **alpha=100** globally. This is a training/validation selection result, not a held-out performance claim. Do not change it after Phase 3.

**[OBSERVATION]** Sampled English-view *character* exposure by arm (means across 5 seeds) is 18.46% (PP), 18.64% (PW), 26.27% (WP), 26.58% (WW). These are source-only training exposures; **do not** confuse them with the theoretical *word-draw* exposure of 23.55% (page allocation) and 32.56% (word allocation).

**[RESULT]** Zero unknown tokens on all 13,760 validation rows. Exactly 344 validation windows per model, 104 English-view and 240 isiXhosa-view. Train and validation have zero overlapping publication issues. No train-to-val exact 13-word shingles survive in the chosen windows.

**[VERIFICATION]** Independent program re-read the original source ZIP (only 1885/1887 payloads), reconstructed all validation windows and 545 LM training documents, checked all 40 model fingerprints, and independently recalculated all 13,760 validation piece counts and log-probability rows with worst numerical difference 2.27e-13 bits. Four fresh refits reproduced learned vocabularies and scores and all 1,376 checked window encodings.

**[UNKNOWN / LIMITATION]** Downstream transfer across 1888 and 1889-94, causal factor contrasts, issue-level uncertainty, genre confounds, and trained duplicate-free sampling remain untested by Phase 2. Earlier research has already inspected later-period samples, so these are not genuinely unseen external data.

**[NEXT GATE]** Run Phase 3 with the full 40 *frozen* model artifacts and alpha 100. Evaluate F1-F4 on paired held-out 1888 and 1889-94 pages using issue-grouped uncertainty. Phase 4 tests repeated-text sensitivity; Phase 5 handles F5 downstream evidence.
