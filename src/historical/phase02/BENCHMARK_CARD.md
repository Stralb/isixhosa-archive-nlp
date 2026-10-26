# Phase 2 benchmark/dataset card

**Target:** change in mean clipped (max .02) *per-page* lexical word-form rate across synthetic counterfactual periods of the same 1885 corpus. NOT real time series, semantic change, topic change, censorship evidence, public opinion or language-community population parameter.

**Raw source:** `TxtLangFiles1870-1894_TB.zip`, SHA256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`; 1885 XhTxt only, 292 pages, 61 newspaper issues, 366,550 alpha tokens. `XhTxt` is an archival text-view assignment, not word-level human language tagging. Rights/permissions for redistribution of raw text require separate checking; raw texts are NOT bundled in this research ZIP.

| Issue-isolated partition | Imvo issues/pages | Isigidimi issues/pages | Total pages |
|---|---|---|---|
| TRAIN | 29 / 116 | 7 / 56 | 172 |
| DEV | 9 / 36 | 2 / 16 | 52 |
| TEST | 11 / 44 | 3 / 24 | 68 |

The original 200-word candidate pool was based on ALL 1885 pages in Phase1; the 80 benchmark words are train-ranked from that pool. Therefore the Phase2 test partition is simulation-condition held-out but NOT archive-independent, and its word-list screening was historically exposed.

**Simulated labels:** 68 no-change lexical forms, six up and six down synthetic lexical forms. A and B populations are identical unless engineered count substitutions are made in B, preserving page lengths and all unchanged features. Repeated mask draws yield lots of mathematical replicates but very few unique historical issues in TEST; aggregate uncertainty across masks is not uncertainty across historical archives.

**Selection:** six frozen mechanisms (source/page MAR, length/date MAR, unobserved-text-dependent MNAR, whole-issue loss, one complete missing source–page cell, repeated-substring-dependent retention). Underlying word text is used by an *oracle* to generate masks, but not available to the fitted propensity method.

**Privileged audit frame:** includes all lost-page IDs, newspaper/source, page half, issue, month, original page token count and retained status. The true selection probability, hidden lexical proxy, repeated-text flag and missing page word-form outcomes are **not** deployable analyst covariates. The `frame.csv` and binary outputs include those oracle fields for reproducibility; an actual challenge deployment must separate private oracle labels from public analyst metadata.

**Methodology:** observed means, matching, stratification, oracle Horvitz-Thompson, logistic propensity estimated from visible frame, deterministic worst-case [0,.02] bounds and two unvalidated pattern-mixture gamma restrictions. The conditional gamma-interval assumption is actively checked using the complete simulation truth. Issue-level dependence explicitly modeled in an entire-issue missingness scenario. No calibrated per-issue 95% confidence intervals claimed in Phase2.

**Ethics:** no personal user data used; historical documents reflect selective publication, editorial control, digitization limitations, mixed languages, reprinted formulae and imperfect OCR. No inference about the beliefs or identities of original speakers. Model outputs should not be used to label historical individuals, genres, or political stances absent validated context. No native-speaker annotation was acquired.

**Reproducibility:** 4 code scripts, frozen protocol/hashes, complete retained and missing masks, source-frame metadata, known synthetic truth, summary metrics, two independently verified partitions and four plots. Original source ZIP is a separate dependency. Self-tests and clean-room checks are documented.
