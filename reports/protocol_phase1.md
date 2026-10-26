# Phase 1 frozen experiment protocol, Version 1 (2026-10-24)

**Status: frozen BEFORE any new 1888-1894 factorial outcome is generated.** This is a new conditional follow-up to Phase 33, not a statistically independent replication. The prospective hypotheses below were developed with knowledge of Phase 33 and earlier results. Do not represent them as novel/independent theory tests.

## Q / causal estimand

Does the observed effect of sampling a historical newspaper by page versus eligible word position depend on (A) how much English-view versus isiXhosa-view text enters training and (B) which pages within a given view supply that text? Does this factorization reproduce the effect across BPE and Unigram, survive excluding repeated newspaper strings and transfer into held-out language-model BPC?

All causal claims are about intervention on **computational sampling distributions**. The original texts, true language identity, historical culture and social outcomes are NOT randomized. `EnTxt`/`XhTxt` are supplied archive block-view assignments, not expert gold labels.

## Dataset, parsing and splits

- Source: user-supplied `TxtLangFiles1870-1894_TB.zip`; original newspaper archive paper: Schoots et al. (2023). Only IMVO used in this particular layout-intervention follow-up; previous cross-newspaper studies remain separate.
- Training: Imvo 1885 and **January-September 1887**. Validation: Imvo **October-December 1887**. Held-out fixed test: 1888; out-of-time robustness: 1889-1894. Date-group disjoint training/validation. The test years have been examined in past phases; this is a precommitted subsequent test with potential analyst-level reuse bias, not untouched external confirmation.
- Lowercase ASCII `[a-z]+` words, no stemming/modernization, identical across all policies. Do not invent linguistic tags. Fixed tokenizer train budget: 500,000 +/- 30 **alphabetic characters**, sampled in 32-word within-page/view windows (last window may be truncated to budget). Archive views with at least 16 words eligible; held-out scoring on first 128 words of view when >=40 words; consistent page/view inclusion across arms.
- Exact page/date identifiers only from filenames; issue cluster = publication date. Distinguish issue-level uncertainty from five model-seed variability.
- No validation or test text enters tokenizer fit or LM training. Additional validation safeguard: tune only on deterministic 32-word validation windows containing no 13-word consecutive subsequence that also occurs in *training* (same archive view), with up to five evenly spaced such windows per eligible page/view and equal-weight view-level aggregation.

## Full factorial sampler

Let `n_iv` be the eligible word count for view v on page i and `n_i = sum_v n_iv` over eligible views. `N` is number of eligible pages.

Two **view-allocation** distributions:

- page-like `p_P(v) = (1/N) sum_i n_iv/n_i`;
- word-like `p_W(v) = sum_i n_iv / sum_i n_i`.

Two conditional **page-within-view** distributions:

- page-conditional `q_P(i|v) proportional to n_iv/n_i`;
- word-conditional `q_W(i|v) proportional to n_iv`.

Sample `(v,i)` from `p_A(v) q_B(i|v)` then sample a uniform eligible 32-word start within that page/view. The four policies are `PP`, `PW`, `WP`, `WW`. `PP` has the same distribution as legacy Phase 33 P; `PW` as legacy M; `WW` as legacy W, under their respective source-eligibility definitions. `WP` is the missing fourth cell. **All four models are refitted on the same new train split**, not compared naively against older fits.

For each model type BPE/Unigram: vocab size 2,000 (previously validation-chosen in original tokenization study, now held fixed), 5 fixed seed identifiers 441-445 (independent arm-specific random streams), same input budget, SentencePiece identity normalization/character coverage 1.0, same permitted character alphabet and configured vocabulary size (NOT an identical learned subword vocabulary), and matched model-fitting budget. Store input-string SHA256 and generated model fingerprints. Training line order and RNG seed are fixed by code.

## Primary metric and statistical method

`E_v = 100 * total_sentencepiece_count_v / total_alphabetic_characters_v` on the identical held-out text. Lower is more compact. Primary test H-F1 is for the English-view BPE representation contrast averaged over within-view policies:

`Delta_A = 0.5 * [(E_WP - E_PP) + (E_WW - E_PW)]`.

`Delta_B = 0.5 * [(E_PW - E_PP) + (E_WW - E_WP)]` is the within-view-page effect. `Delta_interaction = (E_WW - E_WP) - (E_PW - E_PP)`.

Report both language views, for each model family and each historical period; also report actual training character exposure (which need not equal expected view-draw share), tokenizer unk rate and page/view coverage. Report mean of 5 per-seed effects; paired 5,000-draw issue-date bootstrap with seed fixed `343401`, resampling *the same issues* across all arms (95% percentile interval); in 1889-94 also show six-year cluster sensitivity. Mark all contrasts tested on previously examined years as **conditional follow-up**. Exploratory comparisons are never silently upgraded to confirmatory.

### Precommitted decisions

- **F1 / primary:** On 1888 English-view, BPE `Delta_A <= -0.30` pieces/100 chars, with entire issue-bootstrap 95% CI below 0 and the 1889-94 effect in the same negative direction. Otherwise reject. `>=0.30` is practical effect floor, not a claim of social importance.
- **F2 / first-tier algorithm replication:** Unigram English `Delta_A <= -0.30`, CI upper <0, later-year effect negative. Treated as dependent algorithm replication (same archive).
- **F3 / bilingual trade-off:** On BPE isiXhosa-view text, `Delta_A >0` with 95% CI lower>0 in 1888 and later-year positive direction. Otherwise inconclusive/rejected.
- **F4 / within-view mechanism:** Unigram 1888 English `|Delta_B|>=0.30` with 95% CI excluding zero, with sign retained in later-year evaluation. This is a conditional hypothesis motivated by Phase 33; never call it independent confirmation.
- **F5 / downstream:** Compare four arms under identical raw language-model train/evaluation strings, choose ONE LM smoothing alpha from `{1,10,100}` on duplicate-aware **1887 Oct-Dec validation only**, globally across views/arms. On English 1888, LM-BPC effect aligned with F1 and magnitude >=0.01 bits/character, 95% CI excluding zero, with the same sign in later years. If not: explicitly say downstream benefit **NOT SUPPORTED**.
- **F6 / repetition falsification:** Train additional `PP` and `WW` models with repeated material excluded using *training-only* exact 13-word shingle collisions across distinct TRAINING issues. For a fair comparison, apply both repeated-excluded and unfiltered sampling to the identical set of eligible page-view cells, with both 500k-character budgets and no other change. Robustness requires the 1888 English `PP-WW` segmentation effect to retain the main direction, have point estimate >+0.30 pieces/100 chars, CI lower >0 and same sign later. This is a precommitted diagnostic, not a wholly new independent dataset.

Primary claims require F1; F2-F6 are robustness, mechanisms or downstream gates. Never declare the joint story a success if F1 fails. No hyperparameter tuning after held-out scores. Report seed effects separately; do not pretend 5 seeds create independent language samples. If running multiple inferential claims, retain all results and use Holm-adjusted p-values if making formal familywise significance statements; the preregistered CIs are mainly uncertainty displays and effect-size gates.

## Control for repeated text, leakage and selection

- Compute training-only 13-word repeated shingles **across different TRAINING issue dates**, never from held-out outcomes. If a window contains any such shingle it is excluded in filtered training. The support-matched unfiltered control uses same eligible page/views but not filtered starts. Re-normalize both factorial factors on this common support; report how many pages/views disappear, by language view.
- For validation, exclude 32-word windows containing any 13-gram seen in training. Use identical eligible validation windows for every arm/family; do not over-sample a long view at the expense of a short one.
- Post-evaluation label training/test exact shingle overlap separately as a contamination *sensitivity diagnostic*, not a hidden inclusion rule. If leakage dominates, weaken claims accordingly.
- Test invariants: disjoint train-val issue IDs; no model trained on post-Sep 1887 text; all evaluations paired on same strings; equal alphabetic budget; all four factor distributions sum to 1; deterministic sampling; no unexpectedly high unknown-token rate; bootstrap uses whole issue not individual pages.

## Model selection, contingency and falsification

The tokenizer family/vocabulary/normalization are **fixed a priori** from prior work. For the count bigram LM use alpha grid on validation only, choose the lowest mean equal-view BPC across models (ties favor alpha=10). Do not select on 1888+. A higher-capacity neural experiment is optional in subsequent phase if independently justified and validated; absence cannot be misrepresented as neural evidence. If source access or model compute fails, record a prespecified feasibility amendment **before** evaluating outcomes, never after seeing a favorable result. If significance or the sign fails, report the failure; do not chase other seed/budget settings. No semantic, historical-belief, sentiment or fairness-harm claim without corresponding evidence.

## Completion

This Phase-1 freeze passes only when source-only audits, code tests and independent reproduction pass; the factorial and held-out results are deliberately **NOT YET RUN**. Future Phase 2+ must reference a SHA256 of this protocol and any dated amendment, and create its own result decision register.
