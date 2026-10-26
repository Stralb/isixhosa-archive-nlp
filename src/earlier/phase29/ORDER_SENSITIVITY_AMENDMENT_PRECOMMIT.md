# Phase 29B - genuine cross-word-order sensitivity (prospective add-on, 2026-10-24)

## Why a separate prospective test is necessary
Phase29's Phase26 `char_wb` feature extractor collects within-word character n-grams; it contains no cross-word transitions. Permuting word order necessarily produces identical TF-IDF vectors, so H29a cannot measure discourse sequence dependence. This mathematical property was recognized **AFTER the Phase29A outcomes were computed**, before any phase29B model outcomes. Treat the Phase29A shuffle ablation as a negative-control/sanity check and H29a as INVALID/INAPPLICABLE, not a hypothesis about natural language ordering.

## Question
Do short-range word-order features improve newspaper-source discrimination on issue-held-out 1888 windows beyond a word-unigram bag of forms? This tests the representation's local order sensitivity, NOT cultural or political difference.

## Frozen conditions
Train 1885 and 1887, test 1888, identical fixed Phase26 581 source-corrected 90-token windows. Fixed C=1, class_weight='balanced', solver liblinear, max_iter=500, random_state=26024. No tuning.
- Word unigram TFIDF: analyzer word, ngram_range=(1,1), min_df=2, max_features=30000, token_pattern (?u)\b[^\W\d_]+\b.
- Word unigram+bigram TFIDF: same except ngram_range=(1,2).
- Cross-word character TFIDF: analyzer='char', ngram_range=(3,5), min_df=2, max_features=30000.
Every method fit/evaluate on ordered text and on three deterministic within-window shuffles (seed=29024,29025,29026), preserving token counts and spelling.

## Predefined outcomes and gates
Primary: equal newspaper/issue mean accuracy. Paired issue-stratified bootstrap 3000 draws, seed 29224.
H29B1: word 1-2 model on ordered text exceeds the word unigram baseline by at least 5 percentage points and 95% paired interval has lower bound >0, indicating useful adjacent-word information.
H29B2: word 1-2 model loses at least 5 percentage points when word order is permuted, averaged across the three shuffles, with paired interval lower bound >0.
H29B3: cross-word character model loses at least 5 points under word shuffle with paired lower >0.
Negative control: word-unigram vectors/predictions must be exactly order-invariant apart from floating-point roundoff. Model outputs checked against Phase26 word-unigram probabilities.

## Cautions
Original 1888 issues have been studied repeatedly; outcomes are methodological diagnostics, not new independent confirmation. Word 1-2 features cannot distinguish syntax from newspaper-specific phrase repetition or genre. No specialist labels are invented.
