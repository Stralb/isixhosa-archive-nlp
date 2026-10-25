# Fresh training reproduction checkpoint: 25 October 2026

The code and archived outputs are publicly available in this repository. This note is evidence from a new, clean source-data run, not an assertion that every phase was freshly retrained to completion.

## What was actually rerun

The original `Phase2/scripts/run_phase2.py` was run against the course-provided `TxtLangFiles1870-1894_TB.zip`, with the frozen Phase 1 protocol. A clean new output directory was used, without copying archived models. The run was split across checkpoints using the original `--stop-after` argument to test resume behaviour.

The original `Phase2/scripts/verify_phase2.py` independently reconstructed all validation windows, verified training-only access, and recomputed the validation language-model likelihoods.

| Check | Fresh-run outcome |
| --- | --- |
| Source archive SHA256 | Matched the pinned course archive |
| Frozen Phase 1 protocol SHA256 | Matched |
| Distinct fitted tokenizer models | 40/40 |
| Training sample SHA256 compared with archived run | 40/40 identical |
| Learned vocabulary SHA256 compared with archived run | 40/40 identical |
| Candidate vocabulary size | 2,000 |
| Clean validation windows per model | 344 |
| Selected bigram smoothing | `alpha = 100` |
| Held-out source text read during selection | 0 |
| Model serialization hashes matching across different output directories | 0/40 |

The last line is **not a training disagreement**. The SentencePiece binary encodes the output prefix. The learned `.vocab` files and the deterministic source-sample hashes were identical for all 40 runs.

You can repeat the comparison after a fresh run with:

```bash
python scripts/compare_training_replays.py \
  --reference coursework/isiXhosa_Excellence_Sprint/Phase2/outputs \
  --candidate /path/to/new-reproduction/Phase2/outputs
```

Restore the frozen reference model files first with `python scripts/factory.py audit` if necessary.

## What remains to be verified independently

A fresh Phase 3 held-out evaluation was started after Phase 2. All 40 model/view scoring iterations appeared in its log, but the execution was interrupted while writing the full per-example output. **A completed fresh Phase 3, Phase 4, Phase 5, and Phase 6 replay is not claimed here.**

Separately, the full **saved-output** audit in GitHub Actions restores the archived models and 275 numerical output files, checks model SHA256 fingerprints, pairs 99,720 segmentation and 99,720 LM rows, and recalculates 64 factorial effects. That is a successful reconstruction and scientific consistency check, not an independent fresh training run.

The previously used 1888 to 1894 years are not presented as an untouched external validation period. Neither tokenizer compactness nor count-bigram BPC is equated with human language understanding or social fairness.
