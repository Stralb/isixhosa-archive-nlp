# Reproduce the research

## First, know which check you are running

**Source-only validation, no archival data and no retraining:**

\`\`\`bash
python scripts/check_frozen_source.py
python -m unittest discover -s factory/tests -v
\`\`\`

The original scripts, preregistered source documents and result-summary CSVs
are checked with the source-only command. This is not the same as re-scoring
all 99,720 saved examples.

**Reproduce the original controlled experiment from the source ZIP:**

\`\`\`bash
python -m pip install -r requirements.txt
python scripts/reproduce_all.py \
  --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip \
  --workdir /absolute/path/to/clean-nlp2-run --dry-run

python scripts/reproduce_all.py \
  --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip \
  --workdir /absolute/path/to/clean-nlp2-run
\`\`\`

The \`--dry-run\` option prints the actual original training, evaluation,
repeated-text sensitivity and verification commands. The full run writes
its own models, manifests, score tables and logs outside the Git checkout.
Use \`--resume\` for an interrupted run instead of deleting checkpoints.

**Independent, compact implementation:**

\`\`\`bash
python factory/run_experiment.py \
  --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip \
  --out /absolute/path/to/independent-run --mode full
\`\`\`

See [factory README](../factory/README.md) for the comparison boundary.

## Dataset and environment

Obtain \`TxtLangFiles1870-1894_TB.zip\` from the original course distribution
or an authorised archive provider. It is not included in this public repo.
The original, pinned SHA256 is

\`\`\`
375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5
\`\`\`

The original runs used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3,
SentencePiece 0.2.1 and Matplotlib 3.10.8. Use
\`OPENBLAS_NUM_THREADS=1\` and \`OMP_NUM_THREADS=1\` where necessary.
A GPU is not required.

## How to audit a full replay

After the original runner completes, inspect the work directory:

- \`Phase2/outputs/model_training_manifest.csv\` and
  \`PHASE2_SELECTION_FREEZE.json\` for training and validation choices.
- \`Phase3/outputs/factorial_contrasts.csv\` and
  \`heldout_scores.csv\` for segmentation estimates.
- \`Phase4/outputs/f6_contrasts.csv\` and
  \`support_audit.json\` for repeated-text controls.
- \`Phase5/outputs/bpc_factorial_contrasts.csv\` and
  \`heldout_lm_scores.csv\` for held-out language-model likelihood.
- \`Phase6/outputs/PHASE6_SCIENCE_AUDIT.json\` for the independent
  seven-test audit.
- \`logs/*.log\` for every original training and verification stage.

The saved-model fast audit in the original research package is:

\`\`\`bash
sh Phase6/submission_code/RUN_FAST_CHECK.sh /absolute/path/to/TxtLangFiles1870-1894_TB.zip
\`\`\`

Run that command from inside the fully populated research workspace,
not from a GitHub checkout missing the larger numerical artifacts.
It verifies frozen 40-model outputs and regenerates four figures without
retraining all models.

## Scientific limitations

We deliberately keep unsuccessful or mixed evidence visible. The exact
sampling contrast depends on the historical corpus, the archive's block
assignments and the intrinsic segmentation or count-bigram metric. The heldout
years had already been examined during earlier research. This project
supports careful claims about corpus sampling, not cultural or social
fairness claims.
