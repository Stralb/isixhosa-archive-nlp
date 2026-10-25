# Phase 1 decision and risk register

| Type | Decision / uncertainty | Evidence, necessity and mitigation |
|---|---|---|
| REQUIREMENT | Assignment is an open-ended historically grounded Responsible NLP controlled study; Method 25%, Results 25%, Discussion 30% | Project 2 brief and rubric. Focus on actual evidence, controls, negative results and interpretation; write-up has strict two-page core limit. |
| OBSERVATION | Prior Phase 33 found robust sampling-to-tokenizer differences in BPE/Unigram but non-robust bigram BPC benefit | Prior precommits and independently verified saved results (no claim of new independent test). |
| DECISION | Final question is archive-layout-induced selection, learned representation and limits of downstream consequences | One coherent mechanism directly tied to Responsible NLP, instead of adding unrelated research topics. |
| DECISION | Full factorial: PP/PW/WP/WW | Existing Phase 33 had only three arms; fourth arm distinguishes within-view page weighting from view exposure without assuming causal mediation. |
| SOURCE FACT / LIMITATION | `XhTxt` / `EnTxt` are archive-assigned block views | Cannot assume perfect language annotations, dialect identity or semantic meaning. |
| DECISION | Train 1885 + 1887 Jan-Sep; validate Oct-Dec 1887; test 1888, later replication 1889-94 | Fixed chronological issue separation; all post-1887 test years have previously been studied and hence are conditional, not untouched. |
| OBSERVATION | 342 eligible source pages / 86 issues for training; 48 pages / 12 issues validation | Fresh source-only ZIP audit; no 1888+ text was read during Phase 1. |
| OBSERVATION | 13-word train-to-val overlap exists (about 12.85% Xh; 1.24% En by shingle count) | Dataset repetition affects nominal split independence; restrict hyperparameter validation to windows with no training-overlapping 13-gram. |
| DECISION | 13-word train-issue repeated-string control plus support-matched unfiltered comparator | Tests whether serial boilerplate artificially creates the effect; 335/341 eligible Xh views and 198/254 En views retain clean training windows. |
| DECISION | Five seeds, fixed 500k chars, vocab 2000 BPE/Unigram | Carries forward Phase 33 budget/family and prior validation-selected vocab; no test-driven selection. |
| DECISION | LM alpha chosen exclusively on duplicate-aware validation | Closes previous lack of a separate Phase33 LM selection stage; same LM raw text and compute across tokenizers. |
| UNKNOWN | Human-coded 1885 article-type/topic labels currently unavailable | Genre and political-meaning explanations remain unsupported; no invented labels. |
| UNKNOWN | Independent external historical corpus, untouched test material and original raw OCR unavailable | Limit conclusions to this archive and these conditional retrospective-follow-up tests; avoid publication or social-harm overclaims. |
| HARD SUBMISSION GATE | Final `README.txt`, code, dependencies, 2-page PDF, contribution statement, NeurIPS 2024 questionnaire and Faculty ethics | The brief states missing NeurIPS questionnaire means not marked; completion gated in Phase 6. Deadline 27 October 2026 17:00 Johannesburg. |
