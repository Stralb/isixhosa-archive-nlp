# Phase 4 frozen protocol: What additional evidence identifies archival lexical change?

Frozen before Phase 4 development/test outcome scoring, 2026-10-24.

## Core question
What NEW information about missing historical pages could actually make the sign of a full-digitized-archive lexical-rate change identifiable? Do inventory metadata alone help? How many missing pages would need to be re-discovered and transcribed to support sign statements with NO unverifiable model assumption? This is a value-of-information experiment using established sharp finite-population bounds, NOT a proposed new NLP algorithm.

## Input freeze and data leakage control
Reuse the unmodified Phase3 benchmarks (adjacent actual year pairs; 1884->1885 DEV, 1887->1888 TEST), both NATURAL and INJECTED worlds, 80 forms, six fixed selection mechanisms, 80/120 masks per scenario, and the period, source, and page-half frame. The actual outcome on masked pages is private oracle information: not available for policy selection; revealed ONLY if selected as a hypothetical recovery. No shared identical pages across periods. Do not retune against TEST. New tests are not independent historical corpora; years 1887/1888 have appeared in earlier project phases.

Input hashes (SHA256):
- dev benchmark: 47551f5db490029f563b77833f4d0eaabd0ea4ce5dbdd43771b821348c95a809
- test benchmark: 67b2dec56e8ad7dfc87896aa0742604afac25a60b83df02c9a84de92cacc68c9
- frame.csv: f3ec7f8c954b0152258020a05ee400aa6db1d17f6d865da8d89bbf5f48c6a19a
- counts.npz: f77110dd1813bc5857f82c4d3c5aac3890842843ce1773df3b5fe2439b7c6c70
- original language-view archive ZIP: 375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5.

## External evidence gate
Classify evidence by whether it supplies (a) published issue/date/page existence, (b) image availability, (c) independent transcription or validated word-form counts from a previously masked page, (d) a statistical assumption about the word-form rate. Item metadata, known printing schedules and inventories do NOT supply lexical outcome bounds beyond the existing 0<=Y<=0.02 unless a verified transcript/quantitative linguistic certificate can be aligned to the exact missing page. Merely finding scans of pages already in the source ZIP is not extra identification information. Physical missing-year claims are documented as separate from simulation masks. No web catalog is to be assumed complete without auditing its issue-level coverage.

## Target and identification
Full digitized-archive page-uniform change in the per-page rate Y=min(word-form count/alpha-word-count,0.02), B-A, with N_A,N_B and every page's period already known. This complete frame is OPTIMISTIC information: for a genuinely lost 1886 run, the number and size of pages may also be unknown. Without textual restrictions, exact sharp marginal bounds per feature are:
   L = (sum observed_B)/N_B - (sum observed_A + 0.02*missing_A)/N_A
   U = (sum observed_B + 0.02*missing_B)/N_B - (sum observed_A)/N_A.
Metadata that only repeats known frame IDs and page counts cannot tighten these bounds. These sets are deterministic identification sets conditional on full page population and 0<=Y<=0.02, NOT statistical confidence intervals. Verify sharpness constructively for each feature by filling unknown pages at 0 or 0.02. Joint multiform sharpness is not claimed (word counts share finite tokens).

## Counterfactual text recovery experiment
Mask outcomes using the existing Phase3 six selection designs. Within a mask, hypothetically obtain previously unavailable word-form counts for EXACTLY floor(q*M) missing pages where M is the mask's total missing pages and q in {0,0.05,0.10,0.25,0.50,0.75,0.90,1.00}. Such recovery is explicitly simulated; it is NOT evidence that public websites supplied missing transcripts. Fixed nested outcome-blind audit order for three policies:
- RANDOM: seeded uniform random permutation of missing page IDs;
- WIDTH_GREEDY: ascending 1/N_t inverse period size (prioritize missing pages in the SMALLER period, which reduces identification-set width more per page), seeded ties;
- CELL_REPAIR: first select one missing page from each completely unrepresented period/source/page-half cell if any; then WIDTH_GREEDY on remaining pages. This may restore modeling positivity but does not eliminate MNAR assumptions.
No policy can inspect masked lexical counts or true propensities before choosing pages. Use deterministic SHA256 seeded RNG keyed by split/scenario/replicate/world/policy, no source-date leakage from TEST into tuning.

## Pre-registered metrics
For each split x scenario x world x policy x budget x mask: mean identification interval width (units occurrences per 10,000 page words); proportion of 80 features with L>+0.0002 or U<-0.0002 (strongly identified practical sign); 100% truth containment check; fraction with narrow enough bounds to determine a signed practical change; zero-cell positivity restored where originally absent. Percent of hypotheses with nonzero true rate is an oracle diagnostic only. For any decisional error, halt as a bug; sharp bounds must include truth. Group summaries across masks, report both worlds separately, and TEST cannot replace DEV.

Primary low-cost budget q=0.25. Secondary q={0,.05,.10,.50,.75,.90,1}. Fixed zero-cell recovery diagnostic. Report per-scenario, averages and leading examples. Predeclared illustrative thresholds (NOT p-values or guarantees): assess whether RANDOM at 25% recovers >=10% practical sign statements; and whether WIDTH_GREEDY reduces interval width by >=2% vs RANDOM at equal q. Failure to meet these thresholds is a negative finding, not a reason to change the benchmark.

## Go/no-go criteria
A genuinely new identifiability capability requires: (1) independently verified additional page-level linguistic evidence, or a defendable pre-specified restriction not already encompassed by standard pattern-mixture / partial identification; (2) a material improvement over those established bounds and existing point baselines at comparable coverage with valid assumptions; and (3) separate historical-corpus validation. If only hypothetical audit recovery improves bounds, that is a quantitative VALUE-OF-INFORMATION result, not a novel algorithm nor observed actual archaeological recovery. Without (1), record HOLD/NO NEW METHOD; proceeding to Phase5 should focus on acquisition of independently verifiable source material.

No semantic/cultural claims, no fabricated genre labels, no paper writing. Preserve null outcomes and all provenance caveats.
