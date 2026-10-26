# Phase 6 verified claim ledger (NOT an extended abstract)

Date: 24 October 2026. All numbers below were checked afresh by `Phase6/scripts/audit_phase6.py` against the Phase 2 training manifest, the 99,720 paired Phase 3/5 score rows and the frozen factorial outputs. The confidence intervals shown were independently verified in the earlier phases and are **conditional publication-issue bootstrap intervals**, not across-language or across-research-team guarantees.

## Defensible main result and calibrated language

**Research question.** When historical Imvo Zabantsundu page/view text is sampled differently into matched-size tokenizer training corpora, which aspects of exposure account for tokenization and count-bigram predictive differences?

**Intervention.** Factor A alters the share assigned to archive English/isiXhosa *views*, and factor B changes conditional page selection inside views; `PP`, `PW`, `WP`, `WW` are the four combinations. This randomizes computational sample construction only, **not** original historical documents, human language identity, culture or editorial choice.

- All 40 model fits: two families (SentencePiece BPE and Unigram), 4 arms, 5 seeds; 2,000-piece vocabulary, 499,970-500,030 training letters per fit. English-view training-character share averaged 18.46% under PP and 26.58% under WW; language-like assignment averaged ~18.55% across the two P arms versus ~26.42% across W arms (small implementation differences exist between samples).
- Fixed source years: Imvo 1885 and 1887 January-September for tokenizer fitting and LM training. Validation October-December 1887, excluding validation 13-gram overlaps with training. Test 1888 (52 issues); later-year within-newspaper transfer 1889-1894 (306 issues). There are 2,493 paired page/view examples, including 134 English-view and 208 isiXhosa-view excerpts in 1888.
- **Representation mechanism:** 1888 English BPE allocation contrast is **-1.8775** pieces/100 alphabetic characters (95% issue CI **[-1.9716,-1.7818]**). Unigram **-1.7948** **[-1.9081,-1.6706]**. Both have negative later-year contrasts. Lower means more compact segmentation.
- **Segmentation trade-off:** 1888 isiXhosa-view BPE allocation contrast **+0.4526** **[+0.4068,+0.4974]** pieces/100 characters; Unigram +0.5549. Therefore improving representation efficiency of one archive view can worsen the other.
- **Reprint robustness:** in 1888 English, duplicate-filtered BPE PP minus WW (the FULL policy difference, not isolated Factor A) is **+2.3891** **[+2.1824,+2.5898]**; a matched-support unfiltered control is **+2.0974**. Both are positive in later years. Filtering excludes cross-training-issue exact 13-word repeats on retained support but cannot remove paraphrases or every repeated notice.
- **Downstream main result:** English BPE *isolated allocation* contrast is **-0.0372167 bits/character** **[-0.0423644,-0.0321188]** in 1888; **-0.0251924** **[-0.0266522,-0.0236545]** in 1889-94. Five of five seeds have negative effects. Alpha=100 chosen from {1,10,100} by duplicate-aware 1887 validation before this evaluation. In 1888 the magnitude is about **1.29%** relative to PP mean English BPC (2.89187): modest intrinsic-model improvement, not task accuracy or historical comprehension.
- **Critical counterexample:** BPE whole-policy **WW minus PP** English BPC is -0.01784 in 1888 but **+0.00450** in 1889-94. The later whole policy is slightly worse. Unigram later whole-policy difference ~-0.00038, with an interval crossing zero. Changing overall sampling policy is **not uniformly beneficial**, even though isolated allocation helps.
- **Additional complexity:** in 1888, isiXhosa-view BPE allocation BPC is **-0.00275** (does not mirror the segmentation penalty), whereas Unigram has **+0.01048** (worse). Representation compactness and predictive efficiency need not move together.

## Claims that are NOT established and MUST NOT appear in the write-up

1. Historical isiXhosa people, readers or speakers are being harmed, excluded or discriminated against by the measured effects. No such outcome measured.
2. Historic political, ideological or cultural sentiment trends. No validated article-level cultural/genre/sentiment labels were obtained.
3. Modern isiXhosa performance, human task accuracy, neural language-model results, transformer results or downstream user benefit. Tested intrinsic smoothed count-bigram predictive loss only.
4. General replication on an independent newspaper. The Excellence Sprint controlled factorial uses **Imvo only** and its later issues, not Isigidimi as an independent holdout.
5. Never-before-seen external validation. Earlier exploratory phases already inspected the 1888-94 corpus, even though Phase 1-5 procedures were subsequently frozen before NEW evaluation.
6. Guaranteed zero contamination. The repeated-text control screens exact 13-word shingles and changes the eligible sampling population. Residual short and fuzzy overlap is possible.
7. An optimal smoothing setting universally. Chosen alpha=100 is best among the *precommitted three candidate values*, and occurs on that grid's boundary.
8. Archive `EnTxt`/`XhTxt` labels are gold language tags. They are *provided text-view assignments*.

## Exactly what is new in our contribution

The proposed contribution is **not** a new tokenizer architecture or the generic observation that language mix affects tokens. It is the controlled **historical page-layout -> dataset sampling -> factorial exposure -> tokenizer cost -> intrinsic predictive performance (and counterexample)** chain. The derived contribution is about archive/data-curation *accountability*, not a newly observed change in nineteenth-century culture.

## Suggested research reporting policy

Cite Schoots et al. (2023) for original archive and transcription provenance; Beelen et al. (2023) for historical newspaper collection representativeness and bias; Kudo & Richardson (2018) for SentencePiece; Rust et al. (2021) for tokenizer-performance prior work. State the precise difference from those prior works. Report F1-F6 result and principal counterexample in concise form; disclose that 32 contrasts are exploration/supporting diagnostics, not 32 independently established hypotheses.
