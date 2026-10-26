# Project 2 excellence sprint: from Phase 33 to final submission

Status: **Phases 1-5 complete and independently verified**; phase 6 is not yet executed. The historical Phase 1-33 results are preserved without revision. This is coursework-focused and **not a claim that 90-100% or publication acceptance is guaranteed**.

**Single project question:** How do document-level sampling decisions in a historical bilingual newspaper archive affect English-view/isiXhosa-view training exposure and learned tokenizer representation, what mechanisms explain these effects, and when do they *not* translate into predictive benefit?

**Central responsibility:** Data curation can make one historical text view statistically easier to represent than another. Such differences are measurable, but neither segmentation compactness nor an archive-supplied language tag establishes actual language proficiency, historical sentiment, or social harm.

| Phase | Research task | Required output | Pass/fail gate |
|---|---|---|---|
| **1 — Scientific audit and preregistration** | Recheck brief, Phase 33, data access, splits, duplicate feasibility, factorial identification, test locking | Frozen protocol, train-only feasibility audit, executable sampler, independent verification, decision register | Every source/split/sampler check passes, validation is available, protocol frozen before new test scores |
| **2 — Controlled experiment implementation & validation** | Complete 2x2 sampling manipulation, 5 paired seeds, BPE and Unigram, fixed compute; select any tunable LM smoothing with duplicate-aware 1887 Oct-Dec validation only | Training manifest/hashes, validation logs, model specification, unit tests, formal amendment log if needed | Same training budgets; no leakage; every arm reproducible; no inspection of future-year test metrics for tuning |
| **3 — Frozen factorial evaluation & cross-year replication** | Score same held-out Imvo 1888 and 1889-94 page-views; paired issue CIs, per-seed stability, language-view trade-off, family interaction | Prespecified hypothesis tests, all per-issue results, one cohesive main figure | Complete paired coverage, correct uncertainty, no post-hoc tuning, reject hypotheses that fail |
| **4 — Confound/falsification laboratory** | Train-only 13-word repeated-text exclusion, matched support unfiltered control, publication-cue sensitivity if warranted, train/evaluation overlap screen, worst-case source availability | Repeated-text exclusion contrast, support audit, negative controls | Main effect survives or is honestly downgraded; no label/genre invented; clean exposure budgets |
| **5 — Downstream validity and consequence [COMPLETE]** | Choose one *validation-justified* downstream LM test; identical raw training/eval strings and compute; compare modelling loss, not just token count. Preserve Phase 33 negative findings as baseline | Model-selection record, BPC tests, uncertainty; or explicit infeasibility/negative decision | Do not claim real-world harm, language understanding, or downstream benefit unless demonstrated robustly |
| **6 — Adversarial review, concise paper, submission** | Peer-review every claim; render/review exact two-page extended abstract **only when authorized**; verify plots and supplementary code; prepare contribution statement and required ethics forms | Rubric-linked report, final figures, full code, `requirements.txt`, `README.txt`, ethics appendices, signed responsibilities | A fresh environment reproduces results; 2-page content cap respected; required NeurIPS 2024 and Faculty ethics included; submission before 27 Oct 2026 17:00 (Africa/Johannesburg) |

## Execution rules

- A new experiment must have a question, hypothesis, alternative outcome, controls, source provenance, metric, decision rule and a saved timestamped precommit before held-out access.
- Distinguish the frozen Phase 33 confirmatory analyses from the new conditional follow-up; earlier 1888-94 data have been examined already. Do not claim untouched evaluation or novel culture findings.
- Every failed hypothesis remains in the final register. Never replace a failed experiment with a more favorable metric after seeing results.
- Prioritize quality of **Method (25%), Results (25%) and Discussion (30%)** over accumulating experiments. Introduction and Background are 10% each.
- If a stronger claim fails, use the valid weaker claim; a small clean claim is preferable to a large false claim.
- Do not draft the paper unless the user explicitly requests it. All administrative material can be prepared beforehand.
