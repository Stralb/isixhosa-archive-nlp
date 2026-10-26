# Phase 33: from historical document sampling to model outcomes
Frozen BEFORE phase-33 tokenizers are trained or their held-out performance is examined (2026-10-24).

## Question
In the supplied Imvo newspaper archive, does selecting archival text uniformly by page rather than proportionally to its amount of eligible language text alter (1) the English-view allocation of tokenizer training, (2) held-out segmentation, and (3) held-out language-model BPC? What part of any difference is associated with altered English-view allocation rather than within-language page weighting?

## Data and qualification
Use only user-supplied `TxtLangFiles1870-1894_TB.zip`. Treat `XhTxt`/`EnTxt` as archive-assigned block views, not language ground truth. Newspaper identity never stands in for speaker/dialect/culture. Training uses Imvo years 1885 and 1887; 1888 is the designated immediate held-out evaluation; 1889-1894 is an out-of-period replication cohort. No cross-newspaper time comparison. These archive years have been used in earlier studies, so this is a prospective *conditional follow-up*, not a never-seen external test. All training and evaluation strings tokenize ASCII alphabetic sequences, lowercased; no invented word/phrase labels.

## Intervention
Three BPE-2000 SentencePiece tokenizers per seed, seeds 331 through 335, trained on approximately 500,000 alphabetic characters from 32-word intact windows, each sourced from exactly one page/view; sample views eligible if they have >=16 parsed words; each draw stays within one page/view. Each model has identical budget/configuration and training archive. No changes after seeing held-out performance.

P (page-uniform): sample a page uniformly, then select its view with probability proportional to view's eligible word count, then sample one 32-word window.
W (word-proportional): sample (page, view) with probability proportional to eligible view word count; sample a window.
M (view-share matched): choose En view with the *expected page-uniform En word probability* from the known training corpus, then choose page proportional to eligible words in that view; sample a window. This holds P's approximate En allocation while giving W's within-view sampling distribution. Differences P-vs-M may reflect page weighting; W-vs-M isolates view allocation up to finite-sample variation.
Random streams seeded deterministically; audit actual allocations, model hashes, duplicate-window rates, total chars. BPE `normalization_rule_name=identity`, character_coverage=1.0, nthreads=1, no vocabulary selection on evaluation.

## Evaluation / analysis
For each page and each archive view with >=40 parsed words, score the same first 128 parsed words (or entire view when 40-127 words). This avoids preferentially choosing favorable examples; report coverage and the change in expected label mix due eligibility. Compute pieces per 100 alphabetic characters `E` (lower is more compact) and strict `<unk>` rate by view. Issue group = source/date. For the 1888 IMVO cohort, obtain issue-resampled 95% CIs for paired (P-W), (P-M), (M-W) differences, retaining five fixed seeds. Repeat *without retuning* on IMVO 1889-1894 by issue and by year, to evaluate time transfer. Seed stability and paired issue uncertainty are distinct.

Train a classical smoothed token-bigram LM separately on each fixed tokenizer from IDENTICAL raw 1885+1887 LM training passages (first up to 128 words of each eligible original page/view). Use fixed add-backoff unigram prior with alpha=10; evaluate negative log likelihood in bits per alphabetic character, using fixed `<s>` transitions for every line, identical test strings, no LM hyperparameter tuning on held-out data. This is a computational downstream proxy, NOT a neural language-model comparison. Sensitivity at alpha 1 and 100 is prelisted and labeled sensitivity only, not a way to choose the best alpha. No evaluator word labels beyond source files.

## Hypotheses, gates
H33a: W En-view training share exceeds P by >=4 percentage points in mean actual TRAINING CHARACTERS across five seeds (deterministic intervention check). This is not a novel empirical historical claim.
H33b: P has strictly higher held-out E on En-view text than W (P-W > 0) in 1888, and effect >=0.30 pieces/100 alphabetic chars, with 95% issue-resampled CI lower bound above 0; replication direction in 1889-94 with four/five seeds. Report Xh change, do not assume harmlessness.
H33c: P has higher En-view held-out bigram BPC than W at alpha=10 in 1888, effect >=0.01 BPC, lower issue-bootstrap CI >0, and direction replicates in 1889-94. Zero/negative results are fully valid.
H33d (mediation diagnostic, NOT causal proof): if H33b holds, compare W-M and M-P to determine whether view allocation accounts for most of observed E difference; no numeric acceptance target because multiple mechanisms could be implicated. Same for BPC. Do not assert a causal mediation estimand without independent manipulation of all underlying covariates.
Negative controls: no training/test page overlap; no target text in tokenizer/LM fit; exact same held-out strings for all tokenizers; identical LM raw training text; no OOV hidden; compare 1888 vs later years; report zero-denominator and sparse En-view issues. If any pipeline validity test fails, stop and repair BEFORE test evaluation.

## Decision
Do not promote a tokenizer inefficiency as social harm or cultural misinterpretation. If H33b/c fail, conclude archive sampling can affect language exposure without demonstrated model-performance consequences under the tested tokenizer and LM. Preserve all negative results and original Phase32 experiment unmodified. No report/abstract writing until user asks.
