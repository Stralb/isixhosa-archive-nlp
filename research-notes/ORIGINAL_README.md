# Historical isiXhosa Responsible NLP — Excellence Sprint

This portable handoff contains **Phases 1–5 completed and independently verified** of a six-phase coursework excellence sprint. Phase 5 (downstream prediction) has completed and passed independent verification; Phase 6 (submission/peer review) has NOT been executed. No extended abstract/paper has been drafted or human language annotation invented.

## The scientific question

How do page-like versus word-like sampling decisions in a historical bilingual newspaper archive change the allocation and selection of archive-assigned English/isiXhosa training content, the resulting BPE/Unigram segmentation efficiency, and the connection to downstream model behavior?

- Phase 1: scientific freeze of 2x2 view allocation x within-view selection; train-only feasibility and duplicate-leakage audit.
- Phase 2: 40 tokenizers (BPE and Unigram x PP/PW/WP/WW x five seeds), duplicate-aware validation; smoothing alpha=100 chosen on 1887 validation only.
- Phase 3: same held-out Imvo 1888 and 1889–1894 text scored by all 40 models; F1-F4 conditional, issue-bootstrap gates passed with documented seed heterogeneity.
- **Phase 4: preregistered F6 repeated-text falsification**, 20 newly fitted BPE tokenizers with identical surviving page/view support for filtered and unfiltered conditions. F6 passed; 1888 English filtered PP-WW +2.389 [+2.182,+2.590], later +2.119 [+2.050,+2.185]. Independent source reconstruction and full clean retraining reproduced all outputs.

Important: historical years 1888–1894 have been examined elsewhere in the research programme; these are **conditional follow-ups, not pristine external confirmations**. `EnTxt`/`XhTxt` are block views assigned by the supplied archive, not gold language annotations. Subword compactness is not evidence of cultural beliefs or social harm.

- **Phase 5: preregistered downstream F5 supported**: BPE English-view allocation -0.03722 BPC in 1888 and -0.02519 later (count bigram LM with global validation-selected alpha=100). Direct full-policy word/word versus page/page gains are not reliably positive later; do not collapse the factorial design.

## Resume

Read `CURRENT_STATUS_PHASE5.md`, `Phase5/PHASE5_GATE.md`, `Phase5/RESEARCH_DECISION.md`, and `Phase5/PHASE6_START_PROMPT.md`. The remaining Phase 6 is a peer-review/submission audit; it should not retrain models or write the paper until explicitly requested.

## Reproducibility

Requires the **original supplied newspaper file** `TxtLangFiles1870-1894_TB.zip` (29 MB), deliberately NOT duplicated in this handoff. Its SHA256 is `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`. Retain the `isiXhosa_Excellence_Sprint/` relative folder layout. For source data, see the supplied archive paper (Schoots et al., 2023). Follow `Phase5/README.md` for source-to-result reproducibility and independent verification.

## Coursework gate

The Project 2 assessment's main extended-abstract content is limited to two pages. The final work needs full code, `README.txt`, `requirements.txt`, a contribution statement, the NeurIPS 2024 ethics questionnaire and the Faculty AI Ethics statement; absent NeurIPS questionnaire means the assignment is not marked. Final deadline: 27 October 2026 at 17:00 Africa/Johannesburg. Phase 6 handles submission packaging once writing is explicitly requested.
