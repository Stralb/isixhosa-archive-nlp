# Phase 2 completion gate — PASS (2026-10-24)

1. **Protocol lock:** original Phase 1 preregistration SHA256 verified; not amended.
2. **Source integrity:** original user-supplied archive SHA256 verified; only training and validation text payloads read.
3. **Data:** 342 training pages (86 dates), 48 validation pages (12 dates), no issue overlap. Within-view 13-word shingle leakage excluded before alpha selection.
4. **Factorial design:** PP, PW, WP, WW x BPE/Unigram x seeds 441-445 = 40 fully trained models. Each 499,970–500,000 alphabetic characters; each vocab exactly 2,000; valid exposure logs and SHA256 model/input records.
5. **Fixed LM:** 545 identical raw training excerpts used for all model families/arms/seeds; 344 identical clean validation windows per model.
6. **Selection:** single global smoothing alpha 100 chosen strictly from duplicate-aware 1887 Q4 validation (alpha 1, 10, 100 compared and all retained).
7. **Numerical verifier:** 40 model fingerprints, 13,760 encodings, 13,760 * 3 log-probability values checked; max residual 2.27e-13. Zero unknown tokens. Source-only code deliberately skips 1888+ content.
8. **Fresh refitting:** 4 new models trained in a temporary fresh location; learned vocab/weights and 1,376 validation encodings were semantically identical.
9. **Test boundary:** zero new 1888-1894 metrics examined; no post-test tuning and no promotion of validation signals into confirmatory claims.
10. **Next step:** Phase 3 permitted after this dated alpha/model freeze. Phase 4 repeated-exclusion models are NOT included yet.

**Final decision: PHASE 2 PASSED. Phase 3 has not started.**
