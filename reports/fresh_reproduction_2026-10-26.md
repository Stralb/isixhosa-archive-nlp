# Certified fresh end-to-end reproduction — 26 October 2026

**Verdict: PASS for the authoritative six-phase coursework experiment.** A separate clean workspace was populated with the **byte-identical original Phase 1–6 source scripts and frozen protocols from this GitHub repository**. The user-supplied class dataset was verified, 60 tokenizers were fitted anew, and the saved scientific outcomes were regenerated and independently audited. This execution took place in a local isolated compute environment, **not on a GitHub Actions runner**; GitHub stores this audit certificate and the source code.

## Source and environment

- GitHub base commit inspected: `5283e72ef5b8324ab4c85cbca316e72d694e50e2`.
- Git checkout was unavailable from the execution environment (network/DNS restriction). Original scripts and protocol files were extracted from the supplied **Phase 6 research handoff** and verified against the GitHub API's current-main Git blob identities: **23/23 exact matches** (20 Python files and three frozen protocol documents).
- Authorised local input: `TxtLangFiles1870-1894_TB (1).zip`, **SHA-256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`**, exactly the hash required by the frozen protocols and `src/reproduce.py`. The separate raw-page ZIP was supplied but is not needed for this language-view study.
- Python 3.13.5, numpy 2.3.5, pandas 2.2.3, sentencepiece 0.2.1, matplotlib 3.10.8; these are the repository's pinned packages.
- All generated artifacts went to a **new external workspace**, not the checkout. No previous archive models, scores or raw newspaper text were copied into that training workspace.
- A tool execution limit interrupted the first Phase 2 invocation after 29 **freshly fitted** models. The exact original training script resumed its **own new-workspace checkpoints**, verified their training sample/model hashes, then completed the remaining 11 models. This was not reuse of the original published checkpoints.
- The run used the stage entry points and arguments documented by `src/reproduce.py`: **Phase 2 train + verify; Phase 3 evaluate + verify; Phase 4 fit + evaluate + verify; Phase 5 evaluate + verify; Phase 6 audit + figures**. All **11/11 commands passed**.
- Additional checks: Phase 1 source-only deterministic sampler smoke test **12/12**; independent Phase 2 fresh refitting of **4 models** and **1,376 matching validation-window encodings**; auxiliary Phase 3/4/5 plots and Phase 5 secondary overlap bootstrap regenerated.

## New-workspace results versus the original research release

| Artifact or verification | New result | Comparison with independently archived original |
| --- | --- | --- |
| Phase 2 main tokenizer fits | **40/40**, same training budgets and per-model sampling fingerprints | **All 40 learned vocabularies and all other training-manifest fields identical** |
| Phase 2 validation | `alpha=100` selected using 1887 validation only | **Validation scores and selection JSON byte-for-byte identical** |
| Phase 3 held-out segmentation | **99,720 score rows**, 32 factorial contrasts | **Both CSV files byte-for-byte identical** |
| Phase 4 repetition control | **20/20 newly fitted models**, no forbidden repeated 13-grams; 49,860 scores | **Held-out scores byte-for-byte identical** |
| Phase 5 bigram modelling | **99,720 score rows**, 32 factorial contrasts | **Both CSV files byte-for-byte identical** |
| Phase 5 non-overlap bootstrap | Secondary diagnostic regenerated | **CSV byte-for-byte identical** |
| Phase 6 final audit | **7/7 scientific checks passed**, 64 contrasts recomputed | **Audit scientific values identical**; only source dataset filename differs |
| Phase 6 final figures | Four PNG figures regenerated | **All four PNGs byte-for-byte identical** |

**Binary-model hashes are intentionally not advertised as byte-identical.** All 60 newly fitted SentencePiece protobuf models were parsed and checked against the archived reference. **60/60 become exactly identical after clearing only `trainer_spec.input` and `trainer_spec.model_prefix`**, which embed the execution directory. All learned pieces, probabilities/scores, normalisation settings and remaining model fields match. Corresponding `.vocab` files also match the original. The 40 + 20 model-training manifests differ from the archived versions only in `model_sha256`. This resolves the serialization-path difference without weakening the scientific comparison.

## Original findings reproduced

- BPE 1888 English-view allocation effect: **−1.877537** subword pieces/100 alphabetic characters; 95% issue-bootstrap interval **[−1.971592, −1.781812]**.
- Unigram 1888 English-view allocation effect: **−1.7948** pieces/100 alphabetic characters.
- BPE 1888 isiXhosa-view trade-off: **+0.452589** pieces/100 alphabetic characters.
- BPE 1888 English-view count-bigram allocation effect: **−0.03721665 bits/character**, 95% issue-bootstrap interval **[−0.04236444, −0.03211884]**.
- Reprint-filtered 1888 English-view PP−WW segmentation contrast: **+2.389148** pieces/100 alphabetic characters.
- Negative result preserved: the complete WW−PP policy change **worsened** 1889–94 English-view BPC by **+0.0045033**.

These are the original **conditional follow-up** findings, not new independent confirmation on previously unseen historical years. Archive EnTxt and XhTxt are unadjudicated text views. Count-bigram improvements are not evidence of neural/Transformer gains or cultural fairness.

## Verification and limitations

For a machine-readable certificate, check [the completed fresh-run certificate](../results/fresh_reproduction_2026-10-26.json) and [the output SHA-256 inventory](../results/fresh_reproduction_2026-10-26_sha256.csv). The hashed outputs were produced locally from the supplied original data. The source data and newly trained duplicate model binaries **have not been published to GitHub**, because the dataset is external and redistribution permission has not been established. The repository's existing CI verifies saved artifacts without retraining; this report **does not** imply fresh reproduction ran inside CI. A future GitHub Actions fresh rerun would require authorised access to the pinned source-data ZIP.

The full coursework's **scientific reproduction gate is now passed**. This does **not** complete the course-required two-page paper, 2024 NeurIPS ethics questionnaire, Faculty AI ethics statement or group contribution statement.
