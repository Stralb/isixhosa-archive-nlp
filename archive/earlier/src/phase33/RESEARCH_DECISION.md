# Phase 33 decision — Archival page layout, sampling policy, tokenization and downstream prediction
2026-10-24 | Empirical research complete; original Phases 1-32 preserved. NO PAPER DRAFTED.

## Main question and scientific status
Does choosing historical newspaper text uniformly by **page** instead of proportionally by **eligible word position** alter the archive-English-view allocation of a fixed-budget tokenizer-training set, which words it can represent compactly, and actual downstream language-model likelihood?

**RESULT:** A robust, computationally causal sampling-to-segmentation effect exists in the user-supplied historical Imvo corpus. The policy effect is reproducible in BPE and SentencePiece Unigram, with a cost trade-off between EnTxt and XhTxt. The stronger sampling-to-language-model-performance hypothesis DOES NOT generalise: a small 1888 effect fails the preset >=0.01-BPC gate and reverses in 1889-94; the sign can reverse with fixed model smoothing. This is a meaningful negative rather than a reason to retune.

## Frozen designs and validity gates
- Trained on **390 IMVO pages / 98 issues** in 1885 and 1887. Each original page/view was parsed into lowercased ASCII alphabetic tokens. Newspaper metadata fields are filenames only.
- Three sampling interventions P (equal eligible page), W (eligible word proportional), M (match P's expected English-view allocation but keep W's within-view page sampling). All use the same source archive, `500,000` +/- 30 alphabetic characters, 32-word within-page excerpts, 2,000-token models, five paired seeds 331-335. M is a diagnostic, NOT causal mediation identification.
- 15 independently trained BPE tokenizers in primary experiment; 15 separately trained Unigram tokenizers in a *conditional, prospectively frozen AFTER BPE results* follow-up. Both families use exact same input training lines per seed/policy; verified SHA256 hashes.
- Heldout 1888 pages are evaluated without training leakage; out-of-period replication 1889-94. Same fixed first 128 words of every eligible (>=40-word) page/view for all models. 2,493 page-view evaluation units, 37,395 scores per tokenizer family (74,790 across both). Missing year 1886 not interpolated; Isigidimi not used in this particular intervention.
- An identically raw-text-trained (623 training page/view units) unigram-prior-smoothed count bigram LM for each BPE tokenizer, fixed primary alpha=10. LM-BPC robustness alpha=1 and 100 pre-specified. This is a weak classical LM; it does not justify assertions about pretrained LLMs.
- Paired issue bootstraps, 2,000 replicates, five seeds; 6-year cluster bootstrap for long-term period as sensitivity. These intervals are conditional on the archive and sampler design, not samples of all historical newspapers.

## Prospective BPE hypotheses
**H33a SUPPORTED:** Word-proportional sampling yielded **25.512%** English-view alphabetic training characters versus **18.302%** with page-uniform sampling: **+7.210 percentage points** across five seeds. M yielded **18.335%**, verifying an approximately P-matched English exposure. Actual line duplication was low (roughly 0.25%-0.8%).

**H33b SUPPORTED:** For 1888 English-view evaluation text, P minus W tokenizer cost = **+1.7413 pieces / 100 alphabetic characters**, 95% issue-bootstrap interval **[+1.4860,+1.9771]**, over the preregistered 0.30 threshold. On 1889-1894 heldout Imvo issues P-W was **+1.2579**, interval **[+1.1831,+1.3294]**; 6-year-block interval **[+1.0173,+1.4887]**. All five seeds and every one of the six later years show the same sign.

**Trade-off, same matched text:** On the isiXhosa-view texts P-W = **-0.4594** in 1888 (CI [-0.6176,-0.3161]) and **-0.7371** in 1889-94 (CI [-0.7946,-0.6769]). Thus reallocating representation from XhTxt to EnTxt saves more En tokens but costs more Xh tokens; neither should be called intrinsically fairer. All `<unk>` rates were zero.

**H33c NOT SUPPORTED:** 1888 English-view classical LM P-W = **+0.00996 bits/character**, CI **[+0.00157,+0.01903]**, below strict preregistered >=+0.01 gate even before replication; 1889-94 **-0.00121** CI **[-0.00360,+0.00121]**. The conclusion is also smoothing-sensitive: alpha=1 makes P-W negative in both periods; alpha=100 makes it positive. Xh-view classical LM consistently favours P (1888 P-W -0.0231; later -0.0128). Therefore no robust end-to-end modeling gain was established.

## Prespecified-after-BPE Unigram tests (dependent extension)
**H33B1 SUPPORTED:** English-view P-W segmentation cost **+2.1670**, interval **[+1.9077,+2.4091]**, in 1888; **+1.6025**, interval **[+1.5123,+1.6957]**, in 1889-94. All later years show the same direction. The source exposure effect is not unique to BPE.

**H33B2 NOT SUPPORTED:** On 1888 English-view text the Page-vs-Share-Matched Unigram cost is **+0.6048**, interval **[+0.3660,+0.8278]**, outside the predefined practical equivalence band of +/-0.5. BPE's analogous P-M difference was -0.0476, interval [-0.3070,+0.1896]. Thus exact view allocation alone does not explain all learned-tokenizer effects, and the size of the residual depends on tokenizer family. *Algorithm-by-sampling interactions below were measured after outcomes and are EXPLORATORY.*

**H33B3 SUPPORTED UNDER ITS DIRECTION-ONLY GATE:** Xh-view P-W cost = **-0.0370** in 1888 (CI crosses zero [-0.2019,+0.1124]) and **-0.3068** in 1889-94 (CI [-0.3680,-0.2462]). Do not describe the 1888 Unigram Xh effect as robust.

## Post-hoc diagnostics, NOT prospective confirmation
- Training-corpus word sampling's expected English-view allocation was 32.073%, versus 22.943% for uniform pages; the 9.131-point difference is dominated by newspaper **page 3**, which contributes **+9.11 pp**. Other page positions' contributions approximately cancel. This is a mathematical decomposition of the observational archive structure, not a historical editorial-intent finding.
- In 1888 the Unigram-minus-BPE difference in P-W English cost is +0.426 pieces/100 chars, paired issue interval [+0.228,+0.639], and +0.345 in 1889-94 (interval [+0.278,+0.410]). The P-M cross-family difference is +0.652 (1888), indicating that the within-language page-weighting component does not transfer unchanged across tokenizer families. These were constructed AFTER seeing the family results; report only as exploratory diagnosis.
- Each later year 1889,1890,1891,1892,1893,1894 individually retains a positive English-view P-W segmentation difference for BPE and Unigram; their sizes fluctuate without a prespecified monotonic historical hypothesis.

## Critically bounded interpretation
The underlying `XhTxt`/`EnTxt` archive categories are *block-level assignments*. They have not been independently verified as sentence-level language gold labels. This research does not validate sentiment, ideology, political agency, historical cultural change, publication editorial policy, individual author language choice, measured social harm, or a production LLM's downstream performance. The experiments use corrected archival text, not original raw OCR. Earlier 1888 and later data were studied in related phases, so no entirely untouched or independent external historical corpus is being claimed.

Data sampling effects on multilingual NLP and tokenizer choice are **not novel general phenomena**. Distinctive case-study evidence is an explicit page-layout -> selection-policy -> different tokenizer algorithms -> limited downstream LM chain for nineteenth-century bilingual African newspapers. Context of related work: Wegmann et al. (2025), ACL Findings, https://aclanthology.org/2025.findings-acl.572/ ; tokenizer choice paper (2024) https://aclanthology.org/2024.findings-naacl.247/ ; impresso work (2024) https://doi.org/10.1080/01615440.2024.2344004 ; digitized newspaper collection bias (2025) https://www.cambridge.org/core/journals/computational-humanities-research/article/whose-news-critical-methods-for-assessing-bias-in-large-historical-datasets/58568BB734FE04AC5860342194602A93 . Do NOT call this a novel model or independent culture discovery.

## Final research decision
**Retain** the robust sampling-to-segmentation result and model-family interaction, with year-transfer and strong negative controls. **Reject** the stronger claim that sampling reliably improves actual historical text prediction or corrects a demonstrated social harm. The next decisive publication gate is a qualified external genre/corpus validation or a longer-context neural downstream task with a *prospectively justified*, independent evaluation. Neither can be replaced by endless tokenizer seeds or arbitrary token budgets. No paper until explicitly requested.

## Reproducibility
See `PRECOMMIT.md` and `EXTENSION_PRECOMMIT.md` and their SHA256 files, `scripts/`, `outputs/`, `unigram/`, `INDEPENDENT_VERIFICATION.json`, `EXTENDED_VERIFICATION.json`, four figures, and `POSTHOC_DIAGNOSTICS.json`. Source newspaper texts are excluded from this derivative package; users must supply their original archive ZIP.
