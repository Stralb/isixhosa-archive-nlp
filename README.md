# Sampling and Representation in a Historical isiXhosa Archive

Reproducible NLP Project 2 experiments using historical *Imvo Zabantsundu* material from the isiXhosa Intellectual Traditions (IsiXIT) collection.

**Research question:** How do the allocation of training material between English- and isiXhosa-labelled archive views, and the page-selection policy within each view, affect tokenisation and downstream language-model loss?

This is a controlled corpus-sampling study, not a novel tokenizer. It uses a 2 by 2 factorial design, BPE and Unigram tokenizers, five sampling seeds, 40 primary model fits and 20 repeated-text controls.

## Main findings

| Held-out 1888 allocation contrast | Estimate | 95% issue-bootstrap interval |
| --- | ---: | ---: |
| English-view BPE pieces / 100 alphabetic characters | -1.8775 | [-1.9716, -1.7818] |
| English-view Unigram pieces / 100 alphabetic characters | -1.7948 | [-1.9081, -1.6706] |
| isiXhosa-view BPE pieces / 100 alphabetic characters | +0.4526 | [+0.4068, +0.4974] |
| English-view count-bigram bits per character | -0.03722 | [-0.04236, -0.03212] |

Lower cost is better. The English-view improvement also creates an isiXhosa-view segmentation trade-off. **The negative result matters:** changing the entire BPE policy from PP to WW in 1889 to 1894 slightly worsened English-view BPC (+0.00450). See [Findings](docs/FINDINGS.md) and the [original claim ledger](research-notes/Phase6/CLAIM_LEDGER.md).

![English-view bigram results](results/figures/PHASE6_ENGLISH_LM.png)

## Run it

Install Python 3.13 with the pinned requirements. To verify the saved research without downloading the newspaper:

```bash
python -m pip install -r requirements.txt
make audit
```

**The audit uses native files already in GitHub.** It verifies saved model fingerprints, pairs 99,720 segmentation and 99,720 language-model scores, and independently recalculates 64 factorial point effects. It is not a fresh retraining run.

For original fresh training, obtain the authorised course archive externally and run:

```bash
make reproduce DATA=/absolute/path/to/TxtLangFiles1870-1894_TB.zip RUN_DIR=/absolute/path/to/new-run
```

The factory validates the data checksum and reconstructs the original frozen phase scripts in a fresh workspace. All 40 tokenizer fits were independently replayed, but the downstream phases have not been certified as a complete new run. Details: [Reproducibility](docs/REPRODUCIBILITY.md).

## Project structure

| Directory | Purpose |
| --- | --- |
| [experiments/](experiments/) | Unmodified original Phase 1–6 Python experiments, plus a separately identified independent implementation |
| [protocols/](protocols/) | Frozen Phase 1 design and Phase 4/5 preregistrations |
| [scripts/](scripts/) | One factory interface and independent numerical checks |
| [results/](results/) | **Direct CSV score tables, native model binaries, vocabularies and readable PNG figures** |
| [tests/](tests/) | Deterministic unit tests |
| [docs/](docs/) | Methods, interpretation, reproducibility and ethical limitations |
| [research-notes/](research-notes/) | Original decisions and detailed phase handoff records |
| [data/README.md](data/README.md) | How to obtain the restricted historical source data |

For a short review, start with the [reviewer guide](docs/REVIEWER_GUIDE.md), the [frozen protocol](protocols/Phase1/FROZEN_PROTOCOL.md), [training code](experiments/Phase2/scripts/run_phase2.py) and the [native downstream results](results/Phase5/outputs/bpc_factorial_contrasts.csv).

## Limits and attribution

The main intervention uses one newspaper. EnTxt and XhTxt are archive-assigned text views, not independently human-verified language gold labels. The later years had been inspected during earlier research and are not an untouched external test. Compact tokenisation and count-bigram BPC do not measure cultural understanding, historical meaning or social fairness.

Source newspaper transcriptions are not redistributed because public redistribution permission is uncertain. The repository is the research code and evidence, not the final coursework paper, NeurIPS ethics questionnaire, Faculty AI declaration or contributor statement. Contributor licensing must be agreed separately.

Sources: [Schoots et al. (2023)](https://aclanthology.org/2023.rail-1.7/) and [Kudo and Richardson (2018)](https://aclanthology.org/D18-2012/).
