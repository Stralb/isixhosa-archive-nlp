# Executable reproduction factory

This directory contains readable Python code, not a placeholder. It runs the main
two-factor isiXhosa archive sampling experiment from the original course dataset.

This implementation is **independent of the original frozen Phase 1-6 scripts**.
The full original scripts and saved artifacts remain available in the separate
public-release ZIP until their complete import is finished. Do not present this
implementation as the original prospective run or as an independently untouched test.

## Run the actual experiment

From the repository root, with Python 3.13 and packages from requirements.txt:

    python -m pip install -r requirements.txt
    python -m unittest discover -s factory/tests -v
    python factory/run_experiment.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --out runs/full --mode full

Full mode trains BPE and Unigram for the 4 factorial arms with seeds
441,442,443,444,445, 2,000-piece vocabulary and the fixed 500,000-letter
sampling budget per model. It uses 1885 and January to September 1887 for
training, October to December 1887 for validation, and 1888 to 1894 for
test evaluation. It selects smoothing from 1, 10 and 100 using validation
data only and reports paired issue-bootstrap intervals for both token
pieces per 100 alphabetic characters and count-bigram bits per character.

A quick development exercise uses a reduced 4,000-character budget and only
BPE/seed 441. **Never report smoke metrics as original study results**:

    python factory/run_experiment.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --out runs/smoke --mode smoke --bootstrap 100

Outputs are placed only in the chosen output directory: model binaries,
training_manifest.csv, validation_scores.csv, selection.json,
heldout_scores.csv, contrasts.csv, run_manifest.json.

No historical corpus text is included in the repository. Full mode checks the
original source SHA256 before training and refuses to overwrite previous output.

## What has been independently checked?

On the original archival ZIP, the input parser recovered 684 training page
views, 96 validation page views and 2,866 held-out page views. The method
identified 545 language-model training documents, 344 uncontaminated
validation windows and 2,493 eligible held-out page views.

For BPE and seed 441, all four sampled-text hashes matched the original
Phase 2 manifest. Learned vocabularies also matched in the four checked
models. The serialized .model bytes differed because the model output path
is embedded by SentencePiece. This is a partial reproduction check, not
certification of every output or an all-model cold restart.

## Scientifically important limitations

- EnTxt and XhTxt are archival assignments, not adjudicated gold labels.
- This is Imvo newspaper analysis, not claims about historical isiXhosa as a whole.
- The held-out years were inspected during earlier research phases.
- Issue-bootstrap intervals condition on the selected tokenizer seeds.
- Token compactness and smoothed bigram BPC do not measure social fairness.
- Repeated-text controls from the original Phase 4 are not yet integrated in this independent implementation.
