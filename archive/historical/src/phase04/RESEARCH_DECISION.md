# Phase 4 research decision: independent archival evidence and recoverability

## Verdict
**Computational Phase 4 COMPLETE and independently VERIFIED. Novel-method / observational-evidence gate: HOLD.** The experiment quantifies why additional page-level lexical outcomes, not catalogue metadata alone, are necessary to shrink sharp unrestricted bounds. No previously unavailable historical text was actually recovered and validated. A simple existing sharp-bound method already gives the appropriate conditional answer; do not invent an algorithm to claim identifiability without new evidence.

## External evidence finding (source provenance audit)
Public NLSA, CRL and Five Hundred Year Archive (EMANDULO) catalogues document holdings and traceable digitization/transcription routes. The IsiXIT publication reports a wholly missing hardcopy run of Isigidimi for 1886 in its collection. Publicly indexed pages from 1887-88 show the potential for independent transcript verification of selected already-covered material. However: no new independently verified transcript or word-form counts corresponding to any truly missing historical page was acquired in this phase. Phase3 masks are simulated omissions from a known archive, not physically lost pages. Neither an item date nor a publication frequency is a word-form statistic. See EXTERNAL_SOURCE_AUDIT.md. Catalogue holdings descriptions disagree in some date-range details and must not be merged uncritically.

## Study and protocol
Held-out 1887->1888, 588 eligible pages, 80 frozen lexical forms, six archival-loss mechanisms, two worlds (NATURAL actual word forms and INJECTED count interventions), 120 masks each, 3 fixed outcome-blind recovery policies and 8 recovery budgets. Previous 1884->1885 development set uses 496 pages, 80 masks/scenario/world. Full frame and original digitized text are available to the simulator as a private oracle; an auditor sees the hidden text only when its page is hypothetically recovered. Total: 57,600 policy-budget/mask score rows, 4.608 million individual per-form interval checks across development and test. This is NOT 4.608 million independent newspapers or a new historical corpus.

## Held-out injected-world primary result
At exactly 25% of initially missing pages hypothetically recovered (mean originally missing 219.89 pages per mask):

| Policy | Mean interval width, per 10,000 words | Practical sign-call share among 80 word forms |
|---|---:|---:|
| Random page recovery | 112.138840 | 0.000000 (0.000%) |
| Width-optimal period priority | 110.745061 | 0.000104 (0.0104%) |
| Empty-cell repair first | 110.752782 | 0.000087 (0.0087%) |

Preregistered low-cost informativeness gate >=10% sign calls for random recovery is not met. Width priority decreases width by 1.24%, below the preregistered 2% low-cost improvement gate; that particular practical superiority hypothesis FAILS even though the policy is algebraically width-optimal at a fixed count on each mask. Zero-cell restoration is an alternative goal, not the same objective.

Across all six mechanisms at q=0, mean width was 149.231568 per 10,000 and no substantial sign calls were available. At q=0.75, width-priority made 7.09% of word-form sign decisions; at q=0.90, 20.34%. With ALL missing pages recovered, 62.5% of the tested word forms have true finite-page effects exceeding the practical 2-per-10,000 threshold, so that is the attainable overall decision share. The final 37.5% are too small for the preregistered practical threshold even with complete text.

For ZERO_CELL specifically, at q=0.05, the coverage-repair strategy restored observable support to the wholly empty source/page-half stratum on every held-out selection mask; the width-greedy strategy did not. At q=0.25, cell repair restored source-cell support but identified ZERO practical signs in this scenario. Positivity restoration does not imply missing lexical-rate identification.

Natural-world tests and the 1884-85 development data show the same broad information bottleneck; the exact sign-call rates differ. All partial-identification intervals contain their simulated digitized-archive truths. This is a mathematical consequence of valid caps, not a claim of empirical generalization or confidence-interval calibration.

## What we proved, and what we did not
1. Existing Manski finite-page bounds are sharp feature-by-feature, conditional on a complete page census and the known clipped rate range. Catalogue-only evidence gives no new restriction after that optimistic census is granted.
2. In our selection simulations, even a large hypothetical audit may be insufficient to support useful directional claims without stronger, verified restrictions; it quantifies a limit, not an NLP innovation.
3. Reobserving one page from a fully absent stratum repairs observational positivity but does not justify MAR or a conditional missingness model.
4. No real Isigidimi 1886 issue was recovered; this study tests *simulated rediscovery* from a complete 1884/85/87/88 digitized sample. No validated semantic-change or historical-culture labels exist here.
5. Phase3 held-out 1887/1888 and these source issues have already been studied repeatedly in earlier work; no untouched cross-corpus replication.

## Publication go/no-go
Do NOT announce a new PRHI method. Do NOT write a paper solely claiming that generic source-aware weights fail under arbitrary MNAR; statistics already establishes this. A publishable extension requires a real, independent page inventory linked to a complementary transcription or genuine supplementary information about missing outcomes, plus independent replication and superior risk/coverage against published robust-inference methods. NEXT should be provenance acquisition or a deliberately different benchmark, not extra tuning on these same simulations.

Important existing missingness literature: Manski (1990/2003), Imbens & Manski (2004), Little (1993) pattern-mixture analysis, Robins et al. (1994) inverse-probability/AIPW. No priority over this literature claimed.
