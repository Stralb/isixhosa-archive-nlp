# Reviewer guide

All six original phases and the native results are browseable without unpacking an archive.

1. Read the [research question](../README.md) and [Methods](METHODS_AND_VALIDITY.md).
2. Inspect the immutable [frozen Phase 1 protocol](../protocols/Phase1/FROZEN_PROTOCOL.md).
3. Read the original [training](../experiments/Phase2/scripts/run_phase2.py), [held-out evaluation](../experiments/Phase3/scripts/run_phase3.py), and [bigram loss](../experiments/Phase5/scripts/run_phase5.py) scripts.
4. Inspect the native [segmentation contrasts](../results/Phase3/outputs/factorial_contrasts.csv) and [BPC contrasts](../results/Phase5/outputs/bpc_factorial_contrasts.csv).
5. Run `python scripts/factory.py audit`. The 64 factorial point effects are independently calculated from paired original score rows.
6. Read the [claim ledger](../research-notes/Phase6/CLAIM_LEDGER.md), including failures and the negative whole-policy counterexample.

A clean reproduction from the private course dataset is described under [Reproducibility](REPRODUCIBILITY.md). This is a one-newspaper historical corpus-sampling study, not a new algorithm or a demonstration of social fairness.
