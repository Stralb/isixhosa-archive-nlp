# Original experimental code

The six Phase folders contain the **original frozen source scripts** used for the controlled isiXhosa coursework research. Their Python code was moved without editing its scientific logic.

| Phase | Main purpose | Primary script |
| --- | --- | --- |
| [Phase 1](Phase1/scripts/) | Archive audit, sampling feasibility and factorial sample design | [factorial_sampler.py](Phase1/scripts/factorial_sampler.py) |
| [Phase 2](Phase2/scripts/) | Fit 40 SentencePiece BPE and Unigram tokenizers, validate smoothing | [run_phase2.py](Phase2/scripts/run_phase2.py) |
| [Phase 3](Phase3/scripts/) | Held-out tokenisation scores and paired issue bootstrap | [run_phase3.py](Phase3/scripts/run_phase3.py) |
| [Phase 4](Phase4/scripts/) | Repetition and matching robustness, 20 additional model fits | [train_phase4.py](Phase4/scripts/train_phase4.py) |
| [Phase 5](Phase5/scripts/) | Downstream smoothed count-bigram bits per character | [run_phase5.py](Phase5/scripts/run_phase5.py) |
| [Phase 6](Phase6/scripts/) | Independent science audit, claim verification and figures | [audit_phase6.py](Phase6/scripts/audit_phase6.py) |

[independent/run_experiment.py](independent/run_experiment.py) is a separately written readability-oriented implementation. It should not be conflated with the original frozen empirical record.

To execute the frozen stages, use [the factory wrapper](../scripts/factory.py) from the root with the course archive. It creates the original Phase1 through Phase6 relative layout outside this checkout before running the scripts.
