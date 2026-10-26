# Research status and continuation register — 26 October 2026

This register separates **verified repository contents**, **completed scientific experiments**, **preserved-but-not-fully-replayable history**, and **unfinished work**. It does not amend any frozen research protocol or result.

## What the main repository preserves

| Programme | Scientific status | Git coverage | Reproduction boundary |
| --- | --- | --- | --- |
| Main NLP Project 2 coursework, six-phase factorial study | Experiment and scientific audit complete | `src/phase1`–`src/phase6`, frozen protocols, methods, raw paired scores, 40 main tokenizer models plus 20 repetition-control models and all 60 vocabularies, figures | `python src/run.py check` verifies saved results. Full **fresh** end-to-end retraining has not been certified. |
| Earlier original research, Phases 1–33 | Completed exploratory/negative/mixed studies and selection history | `src/earlier/` with original code and decision records; selected compact results under `results/earlier/` | Earlier intermediate datasets, external texts, large arrays and some binary checkpoints are not in Git. Do not treat early results as independent final confirmation. |
| Separate Historical Inference Phases 1–4 | Benchmarks and audits complete; **no independently novel inference method** | `src/historical/`, selected `results/historical/`, [research history](research_history.md) | Large simulation arrays, archival inputs and some intermediate files are external. Phase 2's same-page pseudo-period shortcut invalidates its original proposed method comparison; do not hide it. |
| Historical Inference Phase 5 | **In progress, no authenticated recovered historical pages** | This register and the original external handoff | Evidence acquisition and validation remain outstanding; no completed Phase 5 release exists. |
| Historical Inference Phases 6–7 | Not started | None claimed | Require a validated evidence/novelty gate first. |

The research-history recovery imported **413 original plain files** with a separate [per-file SHA-256 manifest](../results/recovery_manifest.csv). The current repository is **not a byte-for-byte publication of every research ZIP**. It intentionally preserves the primary scientific code and selected readable evidence rather than republishing historical transcriptions, large simulation outputs or encoded transport files.

## Complete external release inventory

The **four user-provided FULL_CHAT_RELEASES archives A–D** contain 29 distinct, nested research ZIP releases. Each nested release's byte count and SHA-256 was checked against the original handoff manifest on 26 October 2026: **29/29 passed**.

The [release inventory](../results/release_inventory.csv) lists every exact archive name, its group, byte count and SHA-256. These packages are **not hosted by Git**; retain the original four ZIP bundles in backed-up, access-controlled storage. The inventory verifies integrity and provenance, **not that all experiments have been rerun**. The main source-data ZIP is also external.

## What is and is not reproduced

1. **Verified:** GitHub Actions passes the unit tests, recovered-file integrity/syntax checks and the saved-model/raw-score scientific audit. The main check recomputes 64 factorial point effects from saved scores.
2. **Partly verified:** the clean replay driver exists and its `--dry-run` is tested in CI. An independent, full source-data-to-final-results training run on this checkout has **not** been certified. Do not describe the CI green status as end-to-end retraining.
3. **External input:** `TxtLangFiles1870-1894_TB.zip` must be obtained through an authorised source and have SHA-256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`. Its supplied copy matched that hash on 26 October 2026. The raw combined-view ZIP is also separately retained, not committed.
4. **Responsible release:** do not add source newspaper text, missing-page scan images, potentially restricted annotations, all ZIP snapshots or raw archived human documents to a public repository before checking rights. No repository-wide code/data licence has yet been chosen; do not infer that the archival text is redistributable from the dataset paper's licence.

## Immediate coursework submission gates (27 October 2026, 17:00 SAST)

The **scientific experiment is complete**; the **coursework submission is not**. The following documents are **not completed in the repository**:

- [ ] Actual extended abstract, final rendered PDF. **Maximum two main content pages** including figures/tables/equations. References and prescribed statements may follow.
- [ ] The course-supplied **2024 NeurIPS ethics questionnaire** from Moodle, with verified answers. The brief says absence prevents marking.
- [ ] Official **Faculty of Science AI Ethics Statement**, including accurate disclosure of AI assistance.
- [ ] Accurate student/group **contribution statement**, with the preliminary two-weeks-prior submission status checked by the group (do not backdate).
- [ ] Confirm the final paper uses the course's expected template/style. `reports/report.tex` is an **unfilled, one-file adaptation** of the provided CCN sample, **not the original `ccn_style.tex`/`ccn.sty`**. Visual/format compliance must be verified before submission.
- [ ] Independently review author names, citations, numerical claims, file access and Moodle upload.

Keep all human statements and institutional forms truthful. Do not fabricate consent, permissions, approval, contribution percentages, or ethics answers.

## Next research stage: Historical Inference Phase 5

This programme is separate from the coursework factorial result. The immediate goal is **evidence acquisition**, not optimisation of a new method. The following five **filename gaps**, not authenticated lost page contents, were verified absent from *both* supplied source-data ZIP inventories:

| Issue date | Candidate page filename |
| --- | --- |
| 1886-03-17 | `IMVO_1886-03-17_p001.txt` |
| 1886-08-18 | `IMVO_1886-08-18_p001.txt` |
| 1887-03-02 | `IMVO_1887-03-02_p003.txt` |
| 1887-04-06 | `IMVO_1887-04-06_p004.txt` |
| 1889-06-13 | `IMVO_1889-06-13_p003.txt` |

A separate year-level Isigidimi 1886 coverage question also remains open; it should not be labelled missing text without archival evidence.

For each candidate, capture repository/shelfmark, issue and *printed* page confirmation, stable archival URL, scan provenance, permissions, transcription method, independent human check, content hash and whether the find adds genuinely **new** authenticated text. Store links and permission-safe metadata in Git; do not upload protected scans or infer words from catalogue records.

**Gate:** only if new authenticated evidence is obtained, preregister a targeted evaluation against the frozen distinct-year Historical Phases 3–4 and published IPW/AIPW/partial-identification/sensitivity baselines. Otherwise close Phase 5 as a documented unsuccessful acquisition. Do not assert novelty or start Phases 6–7 by default.

## Maintenance priorities after submission

- Run and document a **fresh end-to-end** coursework replay from the pinned input ZIP, on a clean workspace, and add the verification log and output hashes only after success.
- Check public code licences and archive reuse/redistribution permissions; choose an appropriate licence only with authority to do so.
- Preserve research files with stable paths and hashes; keep `main` frozen as a reliable baseline. Use new branches, prospective hypotheses and a separate held-out evaluation for future publication-oriented work.

**Integrity rule:** a green CI check is evidence of the tests it ran, not a claim that every older experiment or new historical hypothesis has been independently replicated.
