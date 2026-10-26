# Phase3 benchmark card — adjacent-year historical selection stress tests

**Task:** recovery of finite-archive change in clipped per-page lexical form rate under simulated missing digitized pages. Units: form occurrences / alpha token per page, capped at .02; contrast is page-uniform year B minus year A. **Not** historical semantic truth, social harm or publisher causal effect.

**Provenance:** user-provided original newspaper TXT language-split ZIP, SHA256 375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5. Archives' XhTxt assignment can contain language/view or OCR inconsistencies. We include only issue records with a complete expected 4-page Imvo or 8-page Isigidimi layout. This excludes two partial Imvo 1887 issues. No original raw article text bundled.

| Cohort | Period A | Period B | Total eligible pages |
|---|---|---|---:|
| Development | 1884: 9 Imvo + 21 Isigidimi issues, 204 pages | 1885: 49 Imvo + 12 Isigidimi issues, 292 pages | 496 |
| Phase3 held-out | 1887: 47 Imvo + 12 Isigidimi issues, 284 pages | 1888: 52 Imvo + 12 Isigidimi issues, 304 pages | 588 |

Eighty most frequent candidate wordforms ranked using 1884/1885 development text; no held-out selection. Two worlds: actual archival lexical changes, and a controlled 12-form count substitution into the later period preserving alpha-token totals. Natural lexical outcomes can themselves change and must not be considered known zeros.

Six simulated loss types: source/page composition, visible length/date, concealed lexical-outcome-dependent selection, whole-issue omission, structural Isigidimi back-page zero probability in period B, and loss favoring/or disfavoring material with repeated 13-token strings. `frame.csv` represents an optimistic administrative census of all missing page IDs and word totals; many archives cannot supply it.

Dev 80 draws per selection type/world, TEST 120. Each draw reselects from the SAME set of historical issues, not hundreds of independent historical archives. Method comparison is conditional on that fixed underlying corpus. Unrestricted Manski intervals are arithmetic identification bounds, not statistical confidence intervals. Oracle known-inclusion HT intervals are approximate selection-design 95% CIs, infeasible under unknown MNAR propensity.

Method and decision thresholds are frozen in PRECOMMIT.md. Evaluation uses mean absolute error, practical wrong directional calls, correct-action recall, issue-aware normal CI inclusion under repeated masking, partial-ID effect containment, and wrong-call vs decision-coverage curves.

Public code/data artifact is derived *page-level count statistics*, provenance metadata and simulation masks; underlying newspaper page text remains in separately supplied original ZIP. Publication reuse permissions need verification before external release.
