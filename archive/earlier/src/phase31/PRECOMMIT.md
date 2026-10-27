# Phase 31 — Out-of-training-period context coverage: frozen protocol
Frozen 2026-10-24, before inspecting any new test corpus tokenization outcomes.

## Why this experiment
Phases 26–30 linked newspaper-source training to BPE segmentation and identical-word frequency alignment, but demonstrated neither large practical information loss nor out-of-time generalization. Test whether the saved source-specialist models change the amount of real historical text fitting into a fixed token budget in **other publication years**. Do not claim language change, genre, or social fairness.

## Question and independent test cohorts
- **Early Isigidimi**: all `XhTxt` pages dated 1870–1884 (before the 1885/1887 tokenizer-training excerpts).
- **Late Imvo**: all `XhTxt` pages dated 1889–1894 (after tokenizer training).
- The periods are intentionally DIFFERENT. Estimate within-paper contrasts only; never use between-newspaper level differences as a cultural or chronological effect.
- Page is the sampling unit; cluster intervals at **issue** level. Use all eligible pages without outcome-driven sampling.
- For each page, split on one or more blank lines; select the paragraph with the most ASCII-letter words (earliest in tie), if >=90; use its first 90 lowercased `[a-z]+` surface tokens. Ignore other paragraphs. Pages with no qualifying paragraph are excluded and counted. This paragraph rule is an imperfect article proxy, not expert genre annotation.
- Preserve exact 90-word sequences for evaluation. No training or test text leaves the licensed archive in the research ZIP; cohort extracted by supplied code.
- Reject any evaluation page whose issue date appears in the Phase30 1885/1887 training years; other-years must stay disjoint.

## Models and intervention
- Reuse (NO further fitting or tuning) Phase30's six BPE-1000 frozen `BPE_IMVO_seed{0,1,2}.model` and `BPE_ISIG_seed{0,1,2}.model`, trained with 60,000 alphabetic characters per source and matched hyperparameters.
- Reuse six already-frozen `LINE_PLACEBO_A/B_seed{0,1,2}.model` as **mixed-source balanced controls**: 30,000 training characters of each newspaper in each tokenizer. They were constructed after original Phase30 results, thus not independent confirmatory treatment allocation. Average A/B within seed.
- Count complete-sequence token pieces from `SentencePiece.encode(' '.join(words))`; compare per-word additive token counts as a cross-check; if these differ, evaluate whole-sequence counts and flag the discrepancy (do not silently substitute).

## Primary outcomes and precommitted tests
- **H31a out-of-period domain specialisation**: within each newspaper group, source-specialist model uses fewer pieces per 100 alphabetic characters than cross-source specialist, averaged over the three saved seeds. Require *both* issue-level 95% bootstrap CIs strictly >0; issue draws with replacement nested within source, fixed tokenizer seed pairing. Bootstrap 2000 replicates, RNG 31024.
- **H31b practical context coverage**: In a fixed B=256-piece budget, the newspaper specialist retains >=2 more complete surface words (out of 90) than the other specialist on average in EACH group, with each issue-level paired 95% CI >0. Otherwise reject practical impact at that threshold. The two-word effect-size gate is a chosen practical minimum, NOT literature-established. Also report proportion of windows whose whole 90 words fit under B=256, but treat this as secondary.
- **H31c balanced-source context mitigation**: compare issue-equal mean retained words at B=256 for the two balanced A/B controls with the *two source-specific models*. A deployment rule selecting one single model without knowing the newspaper source incurs two possible worst-group shortfalls relative to the domain specialist; hypothesis is that the balanced models reduce the larger of these two shortfalls by >=50%. Report paired bootstrap CI for benefit. Threshold defined before scoring; if either exclusive model dominates balanced or effect small, reject.
- Secondary robustness at B in {192,224,288,320} pieces, and full-word-retention results. DO NOT pick the best budget post hoc.

## Controls / validity
- Report per-year, source, issue counts; exact issue de-dup; lengths fixed to 90 for all groups.
- All 12 models must have identical 1000-size BPE specifications where applicable (source models and mixed controls); check model file hashes and training audits; no new model training.
- Count `<unk>` in each condition, and report any group with >=0.5% unknown tokens separately; alphabetic-only preprocessing should limit this.
- Issue-grouped uncertainty captures variability in this archive, not population-wide inference; three seeds are matched technical repeats, not 3 independent corpora.
- Results are **new time periods for this specific model-family comparison**, but the same digitized archive has been studied repeatedly. Do not claim an external dataset.
- Historical spelling, source/genre imbalance, and curated language-view labels remain unresolved.

## Predefined decisions
- If H31a passes, retain the narrow claim that newspaper-trained BPE specialization transfers to older/later archive periods within each paper.
- If H31b fails, segmentation benefits are too small to assert large token-window information exclusion under this tested 90-word/256-token setting.
- If H31c fails, do NOT assert that balanced training is operationally preferable to both specialist tokenizers.
- Any statistically interesting subgroup/year discovered later is diagnostic only. No paper drafting.
