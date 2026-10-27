# Phase 29 - prospective representation ablation (2026-10-24)

## Source of hypotheses and status
Follow-up to Phases 26-28 and known failure to access the already-human-coded 1885 article type/topic labels in IsiXIT. This is a **prospectively specified sensitivity analysis on previously used historical issues**, NOT independent confirmation and NOT a genre-controlled study.

## Question
Do held-out source-identification differences require within-passage sequential word order, or can they be explained largely by the distribution of historical surface forms and crude token lengths? Are results driven by identifiable cross-issue recurring passages?

## Frozen data
581 unique 90-token isiXhosa windows in `isiXhosa_phase26_culture_pilots/windows.jsonl`; 1885 and 1887 train; 1888 test. Retain exactly the Phase26 train/test split by year; no tuning/test-based feature selection. Three source classes? No: binary `IMVO` vs `ISIG`; issue balancing at evaluation. Do NOT interpret source identification as political meaning, sentiment, true dialect or human genre labels.

## Models and transformations
A. Full, ordered, source-unmasked Phase26 text (character TF-IDF 3-5 `char_wb`, min_df 2, max_features 30000, C=1 balanced logistic, liblinear, random_state 26024). Refitted from scratch, reproduction check against Phase26 saved predictions.
B. Word order destroyed: three deterministic within-window independent uniform random permutations of the exact same 90 tokens. Both training and test transformed, same classifier as A. No changed spelling, counts, labels, source budget or test issue membership.
C. Word-shape only: bins for token length [1-3,4-6,7-9,10-12,13+], TF-IDF word one-to-three-grams with same LR. Train/test use fixed bins; source word identity unavailable. Additional order-free 5-bin proportions per window with class-balanced LR (C=1).
D. Boilerplate sensitivity: first identify windows with any *exact* normalized 13-token sequence appearing in at least two distinct issue IDs across these windows (no model labels/results used in selection). Refit A after removing these windows in **both training and test**, and compare to A evaluated on the SAME retained test windows. Also apply B to retained windows for a common-sample comparison. No precommitted claim of genre control.

For model fit, fixed default in Phase26: LogisticRegression(C=1, class_weight='balanced',random_state=26024,max_iter=500,solver='liblinear'). Train by 1885 + 1887, no calibration and no validation tuning.

## Metrics
Primary: issue-balanced accuracy, equal weight each newspaper and each issue within it; issue-stratified bootstrap 3000 draws with RNG 29024. Secondary: page-level balanced accuracy, source-specific issue accuracy, model agreement, held-out vocabulary coverage where relevant.
Repeated-issue passages are excluded by unsupervised text signature; counts and removal composition reported.

## Precommitted decision gates
H29a (word-order dependence): >=5 percentage-point absolute loss in issue-balanced accuracy after word shuffling vs A in *each* of 3 seeds, with the paired issue-bootstrap 95% interval for the seed-mean loss excluding zero. If it fails, reject a claim that the existing source classifier needs word sequence order.
H29b (length pattern): at least one of the two predesignated word-shape representations has issue-balanced accuracy >=60%, with lower issue-bootstrap 95% bound >50%. This would reveal a length-only source signature, NOT genre/semantics.
H29c (boilerplate): source prediction persists on boilerplate-excluded test windows if model D issue-balanced accuracy >=65% with lower 95% CI >50%; report difference A_on_retained - D on identical retained cases regardless of direction.
Negative control: 40 train-label permutations stratified within training year; compare resulting source accuracy distribution to originals descriptively. Do not use controls for tuning.

## Leakage and caution
The source texts, period, and test issues were previously investigated in Phases 26-28. The thresholds were chosen before these ablation results, but the data are NOT fresh. Sampling unit is middle of one paragraph per page; source/genre mixture remains uncontrolled. Shuffling words breaks phrase structure but preserves word forms and within-word character grams; equal performance does NOT show semantic equivalence. The archive and language views are expert-corrected texts, not raw OCR. Results may explain measurement properties, not cultural causality.

## Inputs frozen
- `windows.jsonl` from Phase26; its SHA256 will be logged below by the independent verify script.
- Existing Phase26 char-model performance and tokenizer models are **context only**; not a source for training-data selection or hyperparameter changes.
