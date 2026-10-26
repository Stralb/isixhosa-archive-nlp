# Reproducibility

## Verify the native saved results

From the repository root, install requirements and run:

```bash
python -m pip install -r requirements.txt
python scripts/factory.py audit
```

The command checks the saved original 40 model hashes, verifies the original scientific audit, pairs 99,720 segmentation rows with 99,720 language-model score rows, and recalculates all 64 reported factorial point effects. All input files for this check are native files committed under [results](../results). There are no compressed chunks or base64 restoration steps.

These checks reproduce the saved calculations, not a fresh model-training run.

## Run the frozen original experiments

Obtain the original course-provided source archive from an authorised source. Its pinned SHA-256 is:

```
375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5
```

First preview the commands, then run:

```bash
python scripts/factory.py full --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip --workdir /absolute/path/to/new-run --dry-run
python scripts/factory.py full --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip --workdir /absolute/path/to/new-run
```

The factory recreates the original Phase 1 to Phase 6 working layout outside this Git checkout, copies the frozen original scripts and preregistration bytes, and executes original training, held-out scoring, repeated-text controls, downstream language-model evaluation and the independent science audit. It writes isolated outputs and logs, and supports --resume after interruption.

The 40-tokenizer fitting stage was freshly reproduced on 25 October 2026, with identical training-sample and vocabulary hashes. SentencePiece may embed the output path in serialised model binaries, so newly trained model file hashes can change even when learned token pieces agree. A complete fresh replay of later phases is not yet certified.

Run unit tests and native-result validation with `make test`. The second independent implementation is explained in [Independent implementation](INDEPENDENT_IMPLEMENTATION.md).

## Scientific limitations

The archive labels are not independently adjudicated language annotations. The main experiment is from one newspaper, and the later evaluation years were inspected during earlier work. Intrinsic tokenisation and count-bigram BPC do not establish social fairness or a change in historical lexical meaning.
