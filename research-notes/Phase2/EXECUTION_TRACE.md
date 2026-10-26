# Phase 2 execution trace (2026-10-24)

1. Loaded the Phase 1 archive, protocol and freeze hashes before touching the Phase 2 test years.
2. Fixed a single script for all 2x2 samplers, both SentencePiece tokenizer families, and all 5 seeds. No design amendment was necessary.
3. Executed BPE fits 20/20, with saved hashes and validation outcomes.
4. Executed Unigram fits; the initial execution was interrupted after 27/40 completed total fits by an execution time limit. No outcome-driven changes were made. Resumed from hash-verified checkpoints to 32/40 and then 40/40.
5. Applied the predeclared selection rule once all 40 fits were complete: global alpha=100, chosen from duplicate-free Oct-Dec 1887 validation only.
6. Re-read the source ZIP for independent validation, recalculated all 13,760 rows (plus the three alpha scores per row), and obtained exact statistical aggregates.
7. Freshly retrained four representative models in a separate temporary directory to verify repeatable samples, learned pieces, piece scores and 1,376 validation encodings.
8. Locked the model manifest, saved validation scores, selection JSON and SHA256s before any Phase 3 analysis.

No 1888-1894 text content was read for the new Phase 2 work. Archive *filenames* beyond 1887 may be examined for inventory, but no held-out text or performance was used for fitting or alpha selection.
