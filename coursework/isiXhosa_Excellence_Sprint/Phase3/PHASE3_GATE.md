# Phase 3 completion gate

**PASS (2026-10-24)**

1. Matched original source ZIP and original SHA256-frozen scientific protocol.
2. Checked the frozen validation selection: global `alpha=100` remains unchanged, and is reserved for Phase 5.
3. Checked all 40 previously trained BPE/Unigram models by SHA256: no refits.
4. Evaluated exactly the same 2,493 held-out page-view excerpts under every model (1888: 342 view excerpts, 52 issues; 1889-1894: 2,151 excerpts, 306 issues).
5. Independently verified all 99,720 encoding/piece-count outcomes against the original newspaper archive. All 0 unknown tokens.
6. Recomputed all 32 contrasts using two implementations and identical 5,000-draw paired issue resampling. Numerical equality achieved.
7. Four predefined F1-F4 **aggregate** gates met, with within-view Unigram seed heterogeneity retained in the report.
8. Held-out text had been examined in prior research phases. Treat as conditional validation, not entirely independent confirmation.
9. Frozen Phase 4 repeated-text controls and Phase 5 downstream BPC tests remain **NOT EXECUTED**.
10. The user has not asked for a paper; no abstract was drafted.
