# Independent implementation

[run_experiment.py](../experiments/independent/run_experiment.py) is a separate readable implementation of the primary factorial sampling and count-bigram logic. It is not the frozen experiment that produced the main findings.

Run its unit tests with `python -m unittest discover -s tests -v`. It requires the externally supplied original archive for full retraining. Never report its small development smoke mode as a scientific result.
