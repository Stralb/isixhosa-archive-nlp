# Phase 33B prospective-after-BPE follow-up: tokenizer-algorithm replication
Date: 2026-10-24; frozen after main Phase 33 results and before looking at ANY Unigram outcome.

Research question: does the page-versus-word sampling intervention change held-out representation similarly under a *different tokenizer learning algorithm*, using exactly the same sampled training sentences, vocab size, pretokenization and source data?

Design: Reconstruct the already frozen Phase33 training samples for P/W/M, five original seeds 331-335. Independently verify the reconstructed input string SHA256 against Phase33 `training_audit.csv`. Train SentencePiece Unigram 2,000-piece models with identical normalization/character coverage, on identical sampled sentences and alphabetic budget. No evaluation-triggered hyperparameter selection. Evaluate the same 1888 and 1889-1894 page-view windows as Phase33, using per-issue bootstrapping (2,000 draws) and six-year block sensitivity for later evaluation. Record `<unk>` and model hashes.

H33B1: for English-view material, Unigram PAGE-WORD cost >+0.30 pieces/100 characters, with issue bootstrap lower 95% endpoint>0 on 1888, and positive replicate direction in 1889-1894. 
H33B2: the 1888 English PAGE-vs-MATCH cost contrast is practically equivalent within +/-0.5 pieces/100 alphabetic chars, requiring its full 95% issue CI inside [-0.5,+0.5]. This indicates source language allocation can account for the difference within that tolerance, NOT a causal mediation proof.
H33B3: on isiXhosa-view pages, the PAGE-WORD cost is negative in both time cohorts, implying a tradeoff. No magnitude gate.

Classify the follow-up as dependent validation: models use same source texts and five samples as the BPE experiment, and design follows the BPE result. Do not claim independent new-corpus replication or substantive fairness outcomes. Failure, including a reversed sign or rank, remains a legitimate conclusion.
