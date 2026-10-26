# Phase 4: sharp per-form identification and the role of actual recovery

Let A and B be finite, fully enumerated digitized page sets with N_A and N_B pages. For a selected word form define the page statistic Y_i=min(C_i/T_i, c), with c=0.02 and T_i the alphabetic word count. Observed outcomes are available only on set O_t and missing outcomes on set M_t. The target is Delta = (sum_{i in B} Y_i)/N_B - (sum_{i in A} Y_i)/N_A.

If 0<=Y_i<=c is the ONLY restriction on each missing page, the exact marginal identification interval is:
L = S_B/N_B - (S_A + c |M_A|)/N_A;
U = (S_B + c |M_B|)/N_B - S_A/N_A,
where S_t is the sum of OBSERVED Y in period t.

**Sharpness, by construction:** Assign the hidden B values all 0 and the hidden A values all c to attain L; reverse assignments to attain U. As the clipped variable can reach 0 and c, these are attainable extremes for ONE feature (subject only to the available page length, which easily allows at least ceil(c*T_i) occurrences for a single token). Every convex-combination target lies between the extremes in the bounded real relaxation. Discrete count granularity may leave tiny gaps; all observed truth values must nonetheless lie within the interval. Simultaneous attainment of all 80 word-form bounds is NOT asserted because counts jointly share page token budgets.

The interval width is exactly:
U-L = c (|M_A|/N_A + |M_B|/N_B).
It does not depend on observed lexical values. Hence knowing only more issue IDs, page dates, sizes, newspaper titles, catalogue scans of already-observed issues, or historical printing frequency cannot shrink this interval if the frame was ALREADY known. Phase3 already provided an optimistic complete frame, so such metadata are not actually additional identification information.

One NEW verified page transcription replaces that page's unknown Y in [0,c] with an observed value, reducing the identification interval width by exactly c/N_t. At a fixed budget of k recovered pages, a greedy width-minimizing policy should therefore first recover from the period with the smaller N_t, as an exchange argument shows. This is a basic optimal design consequence of known partial-identification bounds, not a novel algorithm. It need not maximize *which specific word forms* become sign-identifiable, since their locations in the interval depend on recovered values.

A page from a previously wholly missing source-page cell restores positive observed support in that cell. But it does not imply the remaining unobserved pages in that cell have the same word statistics. Source-cell-standardized estimators can then *produce* a number conditional on MAR-like assumptions, yet partial identification remains broad. Restoring positivity is NOT the same as repairing outcome-dependent missingness.

A bound that lies wholly above +0.0002 or below -0.0002 is a valid conditional practical sign statement for this fixed finite-page archive and cap. Wrong signs are arithmetically impossible under those assumptions; this is not a statistical 95% guarantee, nor information about lost publications beyond the digitized frame.

The general identification and sensitivity ideas predate this research; this is an empirical value-of-information application of partial-identification theory, not a new theorem. For any real archive without a complete issue/page census, the page frame itself may be unknown, making these optimistic bounds inapplicable without further inventory assumptions.
