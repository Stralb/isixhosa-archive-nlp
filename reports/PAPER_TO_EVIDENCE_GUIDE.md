# Reader's guide: from the two-page Responsible NLP paper to reproducible evidence

> **Revised manuscript and official template sources (26 October 2026):** [CCN-formatted PDF](paper_draft/report.pdf) · [LaTeX and source bundle](paper_draft/) · [full, original CCN and NeurIPS 2024 ZIPs](../templates/original/). The body occupies pages **1–2** and references begin on page 3; declarations are unapproved placeholders. Authorship order: Arno Strauss (2613224), Priyanka Gohil (2586288), Mohammed Bangie (2610990). The integrated three-panel figure combines intrinsic segmentation effects, downstream factorial contrasts and separately calibrated ArchiveLens measurements without conflating their statistical targets.


**NLP Project 2 | 26 October 2026 | Companion to the forthcoming extended abstract.** This is a map of **completed** research, not a substitute for the assignment's required two-page PDF or completed ethical declarations. The PDF source at \`reports/report.tex\` is still a **blank scaffold** as of this guide's creation; do not present that file as a submission-ready report.

## The one coherent question

**How do data-curation and measurement choices affect (i) computational representation and (ii) the reliability of surface-pattern conclusions in digitised historical isiXhosa newspaper text?**

We study two related but **distinct branches** of this same Responsible NLP problem:

1. **Controlled model-representation experiment:** On *Imvo Zabantsundu*, independently vary how the fixed training-text budget is allocated across archive language views and how pages are chosen within each view. Train BPE and Unigram tokenizers; measure signed segmentation cost and smoothed count-bigram loss on held-out dates. This establishes effects **of our controlled sampling interventions on the tested models and texts**.
2. **ArchiveLens claim-sensitivity instrument:** On *Imvo Zabantsundu* and *Isigidimi samaXosa*, choose 120 within-word four-character patterns **from 1887 only**, compare their 1885-to-1888 frequency differences under four alternative weighting estimands, and audit their uncertainty, randomisation reference, influential publication issues, repeated passages and page positions. This is **observational measurement sensitivity**, not a tokenizer-mediated cause of historical language change.

**Crucial distinction:** ArchiveLens did not use tokenizer outputs to define its historical pattern changes. The experiments are connected through *data curation and measurement*, not a proven causal chain from tokenization to historical interpretation.

## Fast path for a marker

| Core paper claim | Main numerical evidence | Full scientific method and verification |
| --- | --- | --- |
| Source exposure changes with sampling unit | In matched IMVO years, 22.643% archive English-view words under equal-page treatment versus 29.475% under word-mass weighting, difference +6.832 percentage points | [Phase 32 decision](../src/earlier/phase32/RESEARCH_DECISION.md) · [sampling estimands](../results/earlier/phase32/sampling_estimands.csv) · [independent replay](../results/earlier/phase32/FRESH_REPLAY_VERIFICATION.json) |
| Sampling interventions redistribute tokenizer efficiency | 1888 BPE allocation: English view −1.8775, isiXhosa view +0.4526 pieces/100 alphabetic chars; the two directions also appear for Unigram | [Frozen study protocol](protocol_phase1.md) · [Methods](methods.md) · [Segmentation contrasts](../results/segmentation.csv) |
| Intrinsic compactness is not sufficient to infer downstream quality | BPE English allocation: −0.0372 count-bigram bits/character; BPE isiXhosa segmentation increases while downstream loss falls slightly on average; many issue-level sign disagreements | [Bigram contrasts](../results/bigram.csv) · [Metric-coupling analysis](exploratory_metric_relationships_2026-10-26.md) · [Issue-level CSV](../results/exploratory/issue_allocation_effects.csv) |
| ArchiveLens surfaces sampling-sensitive **descriptive** trends | 1885-to-1888 change directions reverse for 23/120 preselected patterns in IMVO and 4/120 in ISIG; no confidently supported opposite-direction pair under the original criterion | [Original ArchiveLens plan](ARCHIVELENS_PRECOMMIT_2026-10-26.md) · [Full results](ARCHIVELENS_RESULTS_2026-10-26.md) · [Claims](../results/archivelens/claims.csv) · [Independent verifier](../src/archivelens/verify_independent.py) |
| Warning counts need calibration; estimator sensitivity can be statistically larger | Observed reversal counts not elevated above permutation reference (IMVO 23 versus mean 29.02); 29/120 IMVO patterns show sampling-estimand **magnitude** sensitivity after Holm correction, 0/120 ISIG | [Follow-up frozen plan](ARCHIVELENS_FOLLOWUP_PRECOMMIT_2026-10-26.md) · [Follow-up results](ARCHIVELENS_FOLLOWUP_2026-10-26.md) · [Calibration CSV](../results/archivelens/followup/calibration.csv) |
| Archive structure influences warnings | Among the union of flagged patterns, leave-one-issue-out reverses the baseline temporal direction in 7/32 IMVO and 4/6 ISIG; exact cross-issue 13-word repetition masking removes 10 IMVO original reversal warnings (and one ISIG) | [Attribution](../results/archivelens/followup/attribution.csv) · [Influential issues](../results/archivelens/followup/influential_issues.csv) · [Duplicate sensitivity](../results/archivelens/followup/duplicate_sensitivity.csv) |

All 95% intervals and statistical adjustments must be named with their actual resampling units and assumptions. In particular, the first ArchiveLens 3,000-replicate intervals remain **unadjusted across 120 correlated patterns**; the follow-up Holm-corrected findings concern a separate **permutation sensitivity-magnitude statistic**, not retroactive validation of every original warning.

## Recommended two-page narrative (paper is self-contained)

- **Motivation and research question:** These digitised newspaper sources preserve historical writing but archive view, page, issue and language exposures reflect collection and sampling choices. Can the resulting model behaviour or descriptive historical conclusions be treated as stable?
- **Two methods, carefully separated:** (A) controlled 2×2 language-allocation/page-selection experiment with two tokenizer families, five seeds, fixed budget and date split; (B) ArchiveLens audit of fixed 1887-selected patterns, 1885/1888 source/issue-weighted estimands and calibration.
- **Results:** One decisive bilingual tokenizer trade-off, one actual BPC measure, and one calibrated ArchiveLens result (including negative sign-count result). Put numerical quantities and uncertainty **in the paper**; repository links do not substitute for core evidence.
- **Discussion (highest rubric weight):** Separate *distributional exposure*, *tokenization compactness*, *downstream predictive utility*, and *historical semantic interpretation*. Explain why no single estimator or metric defines fairness, why null tests and repeated-text checks matter, and when the tool should **abstain from linguistic/cultural claims**.
- **Conclusion:** Responsible historical NLP benefits from explicitly recording the exposure estimand, comparing outcomes per archive view, and auditing the sensitivity of descriptive claims to defensible alternative choices.

The existing [representation figure](../results/figures/PHASE6_XH_TRADEOFF.png), [ArchiveLens overview figure](../results/figures/archivelens_claim_sensitivity.png), and [calibration/attribution figure](../results/figures/archivelens_followup_calibration_attribution.png) are source material for **one compact, purpose-built two-panel figure** in the final extended abstract. Do not cram all three figures into two pages.

## What the research does NOT establish

- No expert-validated word-level English/isiXhosa language identity: \`EnTxt\` and \`XhTxt\` are archive-assigned corrected text **views**.
- No human-coded article topic, genre, sentiment, political belief, morphology, historical semantic change, editorial intent or measurable societal harm.
- No true out-of-archive external replication of the **final** 2×2 study or ArchiveLens. Two newspapers belong to the same IsiXIT digitisation collection; earlier book/modern-corpus pilots were different experiments.
- 1888 and the later historical years were viewed during earlier research phases. "Held-out" describes the frozen training/validation/test **split**, not an independently untouched evaluation throughout this larger research programme.
- A sign reversal near zero is not a statistically established contradictory direction. The count of sign reversals was *not* unusual under the chosen year-label permutation reference. The reference assumes exchangeable issue labels within calendar months, which is not automatically a realistic historical null.
- Removing repeated passages changes evaluated support; it is a sensitivity analysis, not a causal estimate of the effects of duplication.
- Standard clustered bootstraps and multiplicity corrections do not establish cultural fairness, correctness of historical interpretations or true generalisation to other sources.

## What is original here, and what is not?

The original methods (BPE/Unigram tokenization, issue/bootstrap estimands, permutation calibration, Holm corrections, exact duplicate masks, leave-one-issue-out influence) are established. The defensible coursework contribution is a **fully reproducible, historically grounded two-part measurement-validity case study** combining (a) a controlled computational sampling intervention with (b) a working ArchiveLens claim-sensitivity and warning-attribution instrument. The strongest insight is *why* different measurements can disagree and why some warnings should be downgraded, not that a general novel fairness algorithm or historical cultural shift was discovered.

Related scholarship: [IsiXIT digitisation](https://aclanthology.org/2023.rail-1.7/), [Schoots on political-root frequencies](https://aclanthology.org/2023.rail-1.8/), [Tokenizer choice and downstream metrics](https://aclanthology.org/2024.findings-naacl.247/), [Bender & Friedman data statements](https://aclanthology.org/Q18-1041/). [Detailed literature comparison](related_work_and_archivelens_proposal_2026-10-26.md).

## Reproducibility in depth

Run with a clean checkout, a verified original source ZIP in \`data/\`, Python 3.13 and pinned [requirements](../requirements.txt):

\`\`\`bash
python -m pip install -r requirements.txt
python -m unittest discover -s src -p 'test_*.py' -v

# Original model study and independent published-contrast audit.
python src/run.py check
python src/audit/independent_ci.py --root .

# Recompute existing model/metric exploratory audit, without retraining.
python src/analysis/metric_relationships.py --root . --verify

# Verify original ArchiveLens outputs from the supplied archive.
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
python src/archivelens/verify_independent.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens

# Independently check the ArchiveLens calibration/attribution extension.
python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify
\`\`\`

For retraining every model rather than checking saved results:
\`\`\`bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
\`\`\`

See [fresh reproduction report](fresh_reproduction_2026-10-26.md), [original independent audit](independent_scientific_audit_2026-10-26.md), [ArchiveLens tool usage](../src/archivelens/README.md), [ArchiveLens follow-up usage](../src/archivelens/FOLLOWUP.md), and [continuous verification workflow](../.github/workflows/ci.yml).

## Responsible use and submission reminder

The project must still submit the actual **two-page extended abstract in PDF**, full code, \`requirements.txt\`, \`README.txt\`, appropriate references, a truthful group contribution statement, the completed **NeurIPS 2024 ethical checklist after references/before supplementary material**, and the **Faculty of Science AI Ethics Statement**. The repository's experimental depth supports the paper but must not be used to evade the strict two-page body limit or academic integrity requirements. Deadline: **27 October 2026 at 17:00 South Africa time**.
