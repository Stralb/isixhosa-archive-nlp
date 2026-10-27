# Earlier research and separate historical inference work

The main Assignment 2 experiment is in [`src/phase1` to `src/phase6`](../src/phase1). Its published results are in [`results/`](../results). This page records the research that came before it and a separate, unfinished publication-oriented investigation. None of these exploratory studies should be treated as additional independent replications of the final coursework result.

The previous phase numbers came from an extended research log. They are useful for finding the original scripts and decisions, but they do not describe one uninterrupted training pipeline. The import preserves the original files where possible, rather than rewriting old experiments to look more conclusive.

## How the coursework question developed

| Research period | What was tested | What we learned |
| --- | --- | --- |
| **1 to 6, early foundation** | Controlled newspaper-source mixtures, tokenizer segmentation and small downstream checks | Changing the training-source mixture changes the measured segmentation gap. This is a source-composition result, not evidence of cultural harm. |
| **7 to 9** | OCR/noise diagnostics, alternate source mixtures, tokenizer merge mechanisms | Some segmentation effects survived source and mechanism controls. More detailed causal claims required stronger controls. |
| **10 to 14** | Neural/count language-model checks, generalisation audits, boundary and robustness certificates | Identified limits of transferring intrinsic tokenizer gains to downstream performance. Some investigations produced null or restricted outcomes. |
| **15 to 18** | Historical retrieval, language routing, difficult queries and transfer | Query construction and language-routing assumptions strongly affected evaluation. |
| **19 to 25** | Nonverbatim retrieval, character retrieval, reprint handling, external text and human-judgement requirements | The apparent retrieval gains depended heavily on synthetic/source-derived queries and how duplicate answers were scored. This line was not accepted as a validated retrieval contribution. |
| **26 to 31** | Newspaper classification, sentiment-lexicon coverage, time transfer, lexical matching, word length, shared-word frequencies and context windows | Several tempting interpretations failed. For example, poor lexicon coverage was not proof of misclassified historical sentiment, and lower token counts did not automatically preserve more context. |
| **32** | English/isiXhosa view visibility under different page and word weighting | How the archive is sampled changes which text is most visible to a tokenizer. This motivated a controlled intervention. |
| **33** | BPE and Unigram sampling-policy pilot with downstream modelling | Segmentation improvements were real but policy effects were mixed, particularly for the complete sampling-policy change. This motivated the frozen final factorial design. |

The surviving early source is under [`archive/earlier/src/`](../src/earlier). Phase 1 to 6 foundations are grouped under `foundation/` because the portable original did not provide one independent code package for every early number. The intermediate packages use their original names, such as `early_transfer/`, `book_ocr/` and `phase12_13/`. Phase 16 contains only the distinctive retained audit/figure code from a rejected draft, not an endorsed publication claim.

For the later diagnostics, open [`phase26`](../archive/earlier/src/phase26), [`phase27`](../archive/earlier/src/phase27), [`phase28`](../archive/earlier/src/phase28), [`phase29`](../archive/earlier/src/phase29), [`phase30`](../archive/earlier/src/phase30), [`phase31`](../archive/earlier/src/phase31), [`phase32`](../archive/earlier/src/phase32) or [`phase33`](../archive/earlier/src/phase33). The most relevant links to the final study are the phase 32 visibility analysis and the phase 33 tokenizer comparison. The selected numerical summaries are in [`archive/earlier/results/`](../results/earlier).

### Important negative results

- The retrieval experiments did not establish performance on independently collected, expert-judged historical information needs.
- The Phase 26 sentiment dictionary covered archival isiXhosa poorly, but there were no adjudicated historical sentiment labels against which to measure misclassification.
- The proposed large cross-source transfer penalty failed its preset effect-size criterion in Phase 27.
- Phase 28 lexical matching was too weak to justify calling the matched passages the same historical topic or genre.
- Phase 30's same-word evidence did not establish the stated improvement on both newspaper populations.
- The Phase 31 256-token context-window test did not show the proposed practically important retained-word gain.
- The final selection-policy experiment showed an isiXhosa-view segmentation trade-off and a later whole-policy BPC counterexample. This is the result developed and audited in the main six-phase coursework experiment.

These outcomes are worth retaining because they explain why the final research question is narrower and better controlled than the first proposals.

## Separate historical inference programme

The research log also contains a different investigation into whether apparent change over time can be identified when historical newspaper pages are missing or unevenly preserved. It is **not** part of the Assignment 2 factorial study, and its numerical outcomes must not be presented as confirmation of tokenizer effects.

| Phase | Status and main lesson |
| --- | --- |
| [1](../archive/historical/src/phase01) | A known-selection simulation and randomisation oracle showed how misleading apparent trends can arise under archive composition. No new estimator was established. |
| [2](../archive/historical/src/phase02) | An unknown-selection benchmark exposed difficult missing-not-at-random cases. A later audit found an important same-page pseudo-period shortcut, which invalidates the earlier proposed method comparison. |
| [3](../archive/historical/src/phase03) | Used distinct historical years and competitive published baseline methods. Risk and calibration transferred poorly under the unknown-selection settings. No new estimator was established. |
| [4](../archive/historical/src/phase04) | Compared hypothetical page recovery scenarios and catalogued archive gaps. The simulations were not authenticated recovery of lost historical text, and their identification limits remained severe. |

The original research scripts and crucial precommits are in [`archive/historical/src/`](../src/historical), with selected aggregated metrics in [`archive/historical/results/`](../results/historical). The Phase 2 shortcut is preserved, not silently corrected away. **Phase 5 acquisition is incomplete**, with no authenticated recovered page text. Phases 6 and 7 were not started. This material is included for research provenance, not as additional coursework results.

## What was restored and what was not

The October 25 handoff contained four consolidated research packages, A through D. Their manifests recorded 29 research releases. We used the original Phase 1 to 33 packages in A and the cumulative Historical Inference Phase 4 release in D. The six-phase Excellence Sprint in B was already represented by the main coursework code; C contains earlier copies superseded by D.

**413 original files** were restored as plain, individually browseable files: **272 Python scripts, 47 Markdown records, 58 JSON records and 36 CSV tables**. The SHA-256 checksums and exact origin paths are in [`results/recovery_manifest.csv`](../results/recovery_manifest.csv). The working import verified every restored file against its original Git blob hash before publication. A small automated test also checks that the published bytes remain unchanged.

This is a **selected code and evidence recovery**, not an assertion that all earlier studies can be cold-retrained from this checkout. Many early experiments require intermediate datasets, additional external texts, original large simulation arrays or binary checkpoints that were deliberately omitted from the portable source package. Individual-text query judgments and archival transcriptions were not republished because permissions and independent labelling were not established. No old ZIPs, encoded payloads, or scanned-book PDFs were added.

The main `python src/run.py check` command verifies the final coursework experiments only. The earlier code is readable and source-preserved, but its historical execution environments and missing artifacts have not all been reproduced. Neither this archive nor the historical inference branch establishes a new, publishable inference method.