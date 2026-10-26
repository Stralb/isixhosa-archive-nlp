# PHASE 26 - PROSPECTIVE THREE-QUESTION PILOT

Created before evaluating Phase 26 outcomes, 2026-10-24. This phase is exploratory and uses the already historically inspected archive; no new confirmatory test claims. Explicit purpose is selection of one culturally meaningful, testable Responsible-NLP question, not production of a paper. Existing Studies 1-4 and Phases 18-25 remain untouched.

Data: supplied, human-corrected isiXhosa XhTxt and EnTxt newspaper page files; archive view labels denote whole textual blocks, NOT gold sentence-level language classes. Grouping unit is newspaper issue. Source documents are not independent observations, and the published archive's genre differences remain confounding factors.

Core preprocessing: lower-case alphabetic words; preserve original historical letters and spelling (no stemming or modern normalization); remove page titles [imvo,zabantsundu,isigidimi,samaxosa], year digits and page-leading metadata by selecting only one paragraph >=85 whitespace tokens per page (choose deterministic middle eligible paragraph) and extracting 90 center words. Discard exact duplicate normalized windows. No label or sentiment is invented. Fixed seed 26024.

Study A - temporal text change:
- Q: Do observable surface-form and character-n-gram distributions differ between 1885 and 1888 within each title, controlling page excerpt length and calendar month?
- H-A: observed JSD for within-word character 4-grams exceeds the 95th percentile of 199 issue-level random time-label permutations performed within source and calendar month. Also show word unigram JSD as a less orthography-specific diagnostic. If sparse monthly groups force changes, log amendment before scoring.
- Effects and limitations: purely distributional textual change, NOT sentiment shift, political agency, author change or population cultural change. Report top term changes as strings without unsupported translations.
- Secondary predefined observed anchors: jabavu, gqoba, makiwane, rulumente; report counts and frequencies without interpreting semantics. No post-hoc cherry picking.

Study B - historic sentiment portability:
- Q: Does a modern English-specific rule/lexicon sentiment analyzer have comparable lexical coverage of the paired newspaper English and isiXhosa views?
- H-B: on same-page bilingual Imvo excerpts (1885/1887/1888), its exact dictionary-hit rate on isiXhosa is <= one third its hit rate on English; report zero-polarity rates separately. Use TextBlob English sentiment dictionary, not a multilingual model. Take pages with >=80 words in both views; 80 alphabetic-word excerpt per view; count matched sentiment lexicon words / 80. Bootstrap by issue. No annotation truth: cannot estimate sentiment accuracy or whether opinion grew more/less negative. Isigidimi data too sparse; report descriptively only.

Study C - publication/domain differences:
- Q: Are there newspaper-source signatures in body isiXhosa passages within the shared 1885,1887,1888 period, even on the entirely held-out 1888 issues?
- H-C: a fixed-character TF-IDF (within-word 3-5 ngrams, min_df=2,max_features=30000)+balanced LogisticRegression(C=1) trained on 1885+1887 predicts 1888 newspaper source with macro balanced accuracy >=.65. Compare with word unigram TF-IDF + same classifier, null=0.5; bootstrap at issue level. Page-title stripping, fixed 90-word windows, issue-held-out test and exact duplicate exclusion. This is a newspaper domain distinction, not identification of political orientation, agency or culture.

Link test to existing tokenizer finding:
- Pre-specified miniature historical tokenizer replication: BPE from matched 150,000 alphabetic characters drawn from 1885+1887 excerpts per newspaper, vocab 1000, fixed 3 seeds, test on 1888 matched excerpt pages. Outcome: pieces/100 alphabetic chars, source-specific source-alignment shift, and macro mean. No model selection on 1888. This is a new diagnostic and must not be mixed with Study-1 confirmatory estimates.

Decision rules: prioritize C+tokenizer only if year-transfer source prediction is robust and source-aligned tokenization shifts similarly; prioritize A only if temporal change survives within-month issue-label permutation and terms can be contextualized without unsupported translation; reject B as substantive sentiment research if no gold-standard validation (coverage findings remain valid). If all fail, stop and record limitations.

All statistical uncertainty clustered at issue level, descriptive means by source not population-level claims. Several tests are exploratory and not independent; do not promote nominal intervals to publication claims. Scripts/results stored separately; no article written.
