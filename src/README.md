# Source code: where to start

**Use `src/run.py` for the coursework experiment.** It is the stable command-line entry point for checking saved scientific outputs or recreating the full original pipeline.

| Location | Role |
| --- | --- |
| [`run.py`](run.py) | User-facing `check` and `reproduce` commands |
| [`phase1/`](phase1/) | Frozen source-only feasibility and sampling design |
| [`phase2/`](phase2/) | BPE/Unigram fitting and validation-only hyperparameter selection |
| [`phase3/`](phase3/) | Held-out segmentation and paired issue bootstrap |
| [`phase4/`](phase4/) | Repetition-filtered tokenizer controls |
| [`phase5/`](phase5/) | Smoothed count-bigram predictive evaluation |
| [`phase6/`](phase6/) | Final scientific audits and figure generation |
| [`archivelens/`](archivelens/) | Independent historical sampling-estimand and sensitivity instrument |
| [`analysis/`](analysis/) | Post-hoc metric/length diagnostics |
| [`audit/`](audit/) | Independent saved-results and confidence-interval checker |
| [`earlier/`](earlier/) | Archived exploratory research stages |
| [`historical/`](historical/) | Separate historical-inference programme, not required for the primary paper |

The exact original filenames for the frozen six-phase experiment are retained because they are checked by provenance manifests and precommitted hashes.

From the repository root:

```bash
python src/run.py check
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay --dry-run
```

See [reproduction instructions](../docs/REPRODUCIBILITY.md) for which operations use the raw archive, and [research map](../docs/RESEARCH_PROGRAMME.md) for the distinction between the two final studies.
