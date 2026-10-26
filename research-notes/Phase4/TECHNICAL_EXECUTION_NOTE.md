# Phase 4 technical execution note (2026-10-24)

The training step fitted all 20 BPE models before scoring held-out text. Writing the combined manifest then raised a CSV `DictWriter` field mismatch because the model metadata included `training_text_included_in_package`, which the manifest header did not. The individual model files and JSON metadata were already saved, with sample/model SHA256 checks. The correction was to add that metadata field to the manifest columns, then rerun the source-only training step reusing and verifying all completed fits. The Phase 1 F6 definition, training sample choices, model settings and test rules were NOT altered. No held-out scores had been calculated before this correction.

The first independent verifier run failed because it zipped a lexicographically sorted support CSV (English first) against the source parser's `xh`-then-`en` iteration order. The verifier was corrected to join by the stable `(filename, view)` key, without changing the training or test data or any scientific outcome.

After an initial successful complete verification, an additional independent check was added to recompute all held-out 13-word train/test overlap diagnostic labels from the original training and held-out source text. This does not alter an outcome or the evaluator, and is recorded as additional QA.
