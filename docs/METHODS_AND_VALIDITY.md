# Methods and validity notes

## Why this question?

Digitised archival collections retain historical editing, page layout, incomplete preservation, transcription choices and researcher sampling choices. The study isolates two choices in *constructing an NLP training corpus*, not causes of the newspapers' historical content.

## The factorial protocol

The Phase 1 freeze specifies a 2 by 2 design. The first factor changes how English and isiXhosa archive views share the available character budget; the second changes selection within a view. The study fits BPE and Unigram under all four conditions `PP`, `PW`, `WP`, `WW` with five seeds, a common 2,000-piece vocabulary and roughly 500,000 alphabetic characters each. All policies share the same training periods, preprocessing and evaluation units. Read the original [`FROZEN_PROTOCOL.md`](../protocols/Phase1/FROZEN_PROTOCOL.md) before adjusting anything.

Model selection used only duplicate-aware windows from October to December 1887. The held-out evaluation uses 1888; later-year transfer checks use 1889 to 1894. The experiment is paired by fixed page/view excerpt. Estimators are evaluated with issue-level bootstrap uncertainty. For repeat robustness, twenty additional models were fitted under a declared matched-support design.

## Threats to validity and what was done

| Risk | Response | Remaining uncertainty |
| --- | --- | --- |
| Imbalanced page and language-view volumes | Fixed character budget and factorial exposure allocation | View differences are not social-group outcomes |
| Page dependence and repeated notices | Publication-issue bootstrap; repeat filter and matched controls | Short or paraphrased repetition can survive |
| Hyperparameter tuning on test | Global alpha selected on 1887 validation | Selection only compares three alpha values |
| Unequal token sequence lengths | Character-normalised BPC instead of token perplexity | Intrinsic bigram model only |
| Archive-specific language extraction | Treat `EnTxt` and `XhTxt` as provided block views | Not validated token-level language ground truth |
| Past exploration of test years | Transparent exploratory lineage and prospective conditional protocol | Not pristine cross-program confirmation |
| Generalisation beyond Imvo | Later issues of the same newspaper evaluated | No independent newspaper transfer claim |

The interpretation concerns what dataset designers can control. It does not attribute the observed distributions to inherent difficulty of isiXhosa or to cultural characteristics of writers.

## Provenance

The initial archive and its human-assisted transcription process are described by Schoots et al. (2023). This repo uses the class-provided historical language-split archive, whose exact SHA256 and data-access instruction are documented in [`DATA_ACCESS_AND_ETHICS.md`](DATA_ACCESS_AND_ETHICS.md).

## Frozen material

Please do not edit the saved model outputs, historical protocols or validation selection lock to obtain a more favourable result. Alternative specifications belong in a new, explicitly labelled analysis. The main confirmatory claims are scoped to the test and bootstrap structure specified above.