# Phase 26: three-question research gate - decision 2026-10-24

## Q1 - Did observable historical newspaper language change?

[RESULT] On the frozen 90-word mid-paragraph excerpts, comparing 1885 versus 1888 by issue, month-stratified permutation detected within-word 4-gram divergence beyond shuffled time labels for **Imvo** (JSD .18763 bits vs 95th percentile .16464, permutation p=.005), but **not Isigidimi** (.19542 vs .19647, p=.115). Word-unigram divergence exceeded permutation baselines for both newspapers (p=.005 each). Each within-title issue has equal weight. There were 49 vs 52 Imvo issues, 12 vs 12 Isigidimi issues. These are actual lexical distribution changes but cannot distinguish historical semantics, topics, genre mix, orthographic variation, or transcription artifacts.

[NEGATIVE] Our prospectively chosen proper-name/government-like strings are too rare in these selected snippets for a cultural/political interpretation (see temporal_anchor_counts.csv). Thus **reject the strong 'political agency evolved' claim**. It is not justified.

## Q2 - Can modern sentiment scores illuminate historical discourse?

[RESULT] On 254 bilingual-page Imvo examples (150 issues) from 1885/1887/1888, an **English-only** TextBlob lexical analyzer had mean exact lexicon hit rate 7.01% on the English-view excerpt versus 0.276% on the isiXhosa excerpt of the same page. Page-level zero-polarity output rates: 5.12% versus 87.01%. Mean paired coverage difference 6.74 percentage points; issue-bootstrap 95% CI [6.32,7.16] points. Eight Isigidimi bilingual pages are insufficient for balanced cross-newspaper sentiment comparison.

[INTERPRETATION] English-only dictionary misses most historical isiXhosa words and returns default-neutral-looking scores. This is a **scope and coverage diagnostic**, not a measured rate of incorrect sentiment labels. A specifically designed or evaluated multilingual isiXhosa sentiment model might behave differently. Without human-validated historical labels, cannot interpret valence as historical attitudes.

## Q3 - Do newspaper environments leave separate linguistic signatures?

[RESULT] Train on 1885+1887 body-isiXhosa excerpts, test on 1888 issues. Frozen char_wb 3-5 TF-IDF+balanced logistic: page-balanced accuracy .7571, issue-balanced accuracy .7640 with issue-stratified bootstrap 95% CI [.7044,.8213]. Word unigram baseline page balanced .7186, issue balanced .7198. All 1888 issues are disjoint from model training issues. 581 unique 90-word excerpts across 1885,1887,1888; 6 repeat windows excluded. 1888: 123 Imvo and 84 Isigidimi pages from 52 and 12 issues.

[POST-HOC ROBUSTNESS] Aggressively remove title variants, political identity-root terms, named historical editors and several common English markers from all train/test excerpts: original frozen setup .7571 page-balanced accuracy becomes .7512. Year-transfer diagnostics from alternative year splits range about .65-.76 depending on split and masks. This test was chosen after seeing primary results and must not count as confirmation. Newspaper domain signatures survive this ablation, but **we cannot infer political ideology or culture from them**; differences in genre, topic, print/OCR practices and editorial contexts remain competing explanations.

## Link to prior tokenizer/disparity and language-model findings

[RESULT - EXPLORATORY DIAGNOSTIC] Train six new, compact BPE tokenizers (vocab 1,000; 60,000 alphabetic training characters per paper per seed; 3 seeds) on same 1885+1887 middle-of-paragraph windows. Held-out 1888 pages favor their matching newspaper tokenizer by **.904 pieces/100 chars (Imvo)** and **.890 pieces/100 chars (Isigidimi)**, averaged across 3 seeds. All 3 seeds had positive savings for both domains. Issue/seed resampling diagnostic intervals Imvo [.074,2.096], Isigidimi [.090,1.419]. Difference D=E(Imvo)-E(Isigidimi) shifts toward Imvo between the two specialists for all 3 seeds (-1.46, -2.18, -1.75). After aggressive named-source masking, all 6 within-source benefits remain positive. This independently computed pilot is compatible with our earlier controlled tokenizer/source-alignment finding but cannot be advertised as statistically independent confirmation: corpus and year 1888 have been repeatedly studied before.

This connection is *operational*: source-specific word/pattern statistics affect both a corpus classifier and BPE segmentation compactness. It does NOT prove that the exact same n-grams causally generate both outcomes or that tokenizer compactness changes semantic interpretation.

## Additional scholarly literature and novelty correction

Jonathan Schoots (2023), **Analyzing political formation through historical isiXhosa text analysis: Using frequency analysis to examine emerging African Nationalism in South Africa**, RAIL 2023, pp.65-75, DOI 10.18653/v1/2023.rail-1.8, explicitly tracks changing political-identity roots (-Ntsundu, -Mnyama) across the same newspapers and editor periods. Our notion of a new 'African agency language evolved' frequency paper would overlap substantial existing published research. We must cite and differentiate from that work. Source: https://aclanthology.org/2023.rail-1.8/

[POST-HOC ILLUSTRATION] Using Schoots's published roots as established scholarly reference points, broad substring root search on 90-word sample windows finds 201 Imvo `ntsundu`-containing tokens across the 3 shared years, versus just 54 exact `abantsundu` forms. However, broad substring matches are *not equivalent* to Schoots's human-referent restricted root rules; counts can include paper-name variants and nonhuman uses. No historical or cultural inference is drawn from that calculation. Root-aware counting itself is not novel here.

[POST-HOC NEGATIVE] Equal-calendar-month, equal-12-issue JSD comparison did not show that paper-vs-paper differences are invariably larger than within-paper year changes: mean Imvo 1885/1888 change .361, Isigidimi .195, cross-title gap .295 (1885) and .318 (1888). Therefore **do not claim newspaper domain dominates historical time**.

## Selection after the three pilots

1. **Prioritize a Responsible NLP *measurement-validity* study**, not a history paper asserting newly discovered ideologies: 'How do historical newspaper source composition, orthographic/morphological variation, and algorithmic preprocessing shape the apparent differences in historical isiXhosa discourse?' It builds naturally on existing source-sensitive BPE and downstream model effects. Use published scholarly root analyses as a *reference case*, not as original research.
2. Do NOT use English TextBlob scores as any finding about historical sentiment; the pilot already establishes why that would be invalid. Modern isiXhosa sentiment datasets, if explored, are still not gold labels for nineteenth-century text.
3. A publishable next gate needs actual *convergent controls*: same-year/topic or article-type matching, morphology-aware term measurement based on validated scholarly roots, and stronger modern isiXhosa-capable baselines. The archive may have 1885 item-level human-created metadata, but these are not contained/aligned with this supplied ZIP. Confirm before claiming topic control.

## Validity boundaries

- All findings are on provided **corrected** texts, not verified raw OCR.
- All three are exploratory and not independent of the many previous 1888 investigations.
- The 1886 Isigidimi gap is real and documented in Schoots (2023); do not interpolate it.
- No isiXhosa expert annotations, sentiment gold labels, historical attitude labels, or causal editorial effects have been added.
- English/isiXhosa archive views are copied text blocks, not perfect word-level LID truth.
- Source classifier can exploit editorial policies, genre, named people, archival formatting, OCR transcription practices or topic mix. High accuracy does not identify which.
- Bootstrap intervals resample sampled *issues* and seeds; they are conditional on this archive and model/data choices, not confidence intervals over communities or historical meanings.
- Results are empirical pilots, NOT yet a publishable novel scientific contribution.
