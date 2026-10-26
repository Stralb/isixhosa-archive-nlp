# Native research results

All results in this directory are **ordinary, individually accessible files**. No archive chunks, base64 snapshots or restore step is required.

| What to inspect | File |
| --- | --- |
| 40 fitted-model metadata and fingerprints | [Training manifest](Phase2/outputs/model_training_manifest.csv) |
| Frozen validation selection | [Selection freeze](Phase2/outputs/PHASE2_SELECTION_FREEZE.json) |
| Held-out segmentation point effects and intervals | [Factorial contrasts](Phase3/outputs/factorial_contrasts.csv) |
| Full held-out paired segmentation score rows | [Segmentation scores](Phase3/outputs/heldout_scores.csv) |
| 20-model repeated-text sensitivity | [Repeated-text effects](Phase4/outputs/f6_contrasts.csv) |
| Downstream smoothed count-bigram effects | [BPC contrasts](Phase5/outputs/bpc_factorial_contrasts.csv) |
| Full held-out paired count-bigram score rows | [LM scores](Phase5/outputs/heldout_lm_scores.csv) |
| Original seven-test scientific audit | [Audit report](Phase6/outputs/PHASE6_SCIENCE_AUDIT.json) |
| Original visualisations | [Four figures](figures/) |

The 40 original tokenizer binaries and 40 vocabularies are in [Phase2/outputs/models](Phase2/outputs/models/). Twenty robustness models and their vocabularies are in [Phase4/outputs/models](Phase4/outputs/models/).

The factory can independently check their byte hashes and recompute 64 factorial point estimates from the underlying 99,720 paired score rows.

Do not interpret compactness or intrinsic count-bigram BPC as social fairness or historical language meaning. See [methods and validity](../docs/METHODS_AND_VALIDITY.md).
