# Historical isiXhosa Archive Sampling and Representation

**COMS4054A/COMS7062A/COMS7066A, Natural Language Processing, Project 2 (2026)**

This repository contains experimental code, frozen protocols, saved result
summaries and checks for a study of how **sampling a historical newspaper
archive changes tokenizer representation and downstream language-model loss**.

The research uses digitised *Imvo Zabantsundu* pages from the isiXhosa
Intellectual Traditions (IsiXIT) collection. It is a reproducible analysis of
NLP dataset construction, not a new tokenizer architecture or a claim that a
historical newspaper represents isiXhosa culture as a whole.

## Start here: run actual code

There are two separate paths:

1. **Original frozen research scripts:** [coursework/isiXhosa_Excellence_Sprint](coursework/isiXhosa_Excellence_Sprint/README.txt). These are the scripts used for Phases 1 to 6, now committed alongside their precommitted designs and selected results.
2. **Independent readable implementation:** [factory](factory/README.md). This offers a compact way to inspect the same 2 by 2 design, with its own unit tests. It is a reproduction attempt, not a replacement for the original record.

You can verify original research files and key result tables without accessing
the source text:

\`\`\`bash
python scripts/check_frozen_source.py
python -m unittest discover -s factory/tests -v
\`\`\`

To run the **original full experiment from the original course dataset**:

\`\`\`bash
python -m pip install -r requirements.txt
python scripts/reproduce_all.py \
  --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip \
  --workdir /absolute/path/to/new-reproduction --dry-run

python scripts/reproduce_all.py \
  --archive /absolute/path/to/TxtLangFiles1870-1894_TB.zip \
  --workdir /absolute/path/to/new-reproduction
\`\`\`

The runner checks the dataset and frozen-protocol hashes, creates a clean
workspace outside this repository, runs the original training, validation,
held-out evaluation, repeated-text sensitivity, downstream bigram experiment
and independent audit, and writes separate logs for each stage. It never
silently overwrites your original research folder. Full retraining needs a CPU
environment with sufficient time and memory.

The original dataset is **not** included. It must be obtained through the
course or authorised archive access. SHA256:
\`375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5\`.

## Research question

When the amount of tokenizer-training text is held approximately constant,
how do two different corpus construction choices affect segmentation and
simple language modelling?

The study separates:

- **View allocation:** whether English and isiXhosa archive text views get
  training exposure according to pages or word counts.
- **Within-view page selection:** whether individual source pages are
  selected with page-like or word-like weighting.

This gives four training policies, \`PP\`, \`PW\`, \`WP\` and \`WW\`. The
factorial contrast distinguishes the effect of the first choice, the second
choice, their interaction and the overall \`WW - PP\` policy difference.

## Experimental design

| Component | Original frozen design |
| --- | --- |
| Training text | Imvo 1885 and January to September 1887 |
| Validation text | October to December 1887, repeat-aware windows |
| Held-out evaluation | 1888, plus within-newspaper comparisons from 1889 to 1894 |
| Main tokenizer fits | 2 algorithms x 4 arms x 5 seeds = 40 |
| Tokenizers | SentencePiece BPE and Unigram, 2,000 vocabulary items |
| Per-model training budget | Approximately 500,000 alphabetic characters |
| Downstream check | Smoothed count-bigram language model, validation-selected alpha 100 |
| Robustness | 20 additional BPE fits for repeated-text / matched-support controls |
| Units for uncertainty | Paired issue-level bootstrap, model seeds held fixed |

The main study includes 2,493 eligible page/view excerpts across 358 issues.
The archived evaluation generated 99,720 paired segmentation score rows and
99,720 language-model score rows.

## Main findings, including the negative result

| Held-out 1888 outcome | Mean effect | 95% issue-bootstrap interval |
| --- | ---: | ---: |
| English-view BPE pieces / 100 alphabetic characters, allocation | -1.8775 | [-1.9716, -1.7818] |
| English-view Unigram pieces / 100 alphabetic characters, allocation | -1.7948 | [-1.9081, -1.6706] |
| isiXhosa-view BPE pieces / 100 alphabetic characters, allocation | +0.4526 | [+0.4068, +0.4974] |
| English-view BPE bits per character, allocation | -0.03722 | [-0.04236, -0.03212] |

A negative allocation effect means the model needed fewer pieces or lower
bits per character under word-like rather than page-like allocation.
The English improvement comes with an isiXhosa-view segmentation trade-off.

**A counterexample matters:** on the later 1889 to 1894 period, changing
the entire BPE policy from \`PP\` to \`WW\` made English-view BPC slightly
worse, with a reported \`WW - PP\` contrast of **+0.00450 BPC**. Therefore
the data do not show that word-proportional sampling is always better.

Read [Findings and negative results](docs/FINDINGS.md),
[Methods and validity](docs/METHODS_AND_VALIDITY.md) and the
[original Phase 6 claim ledger](coursework/isiXhosa_Excellence_Sprint/Phase6/CLAIM_LEDGER.md).
The ledger distinguishes evidence, interpretation and claims we must not make.

## Find the code and evidence

| Path | What to examine |
| --- | --- |
| [Phase 1](coursework/isiXhosa_Excellence_Sprint/Phase1) | Frozen design, corpus audit and factorial sampler |
| [Phase 2](coursework/isiXhosa_Excellence_Sprint/Phase2/scripts) | 40 tokenizer fits, validation selection and independent replay |
| [Phase 3](coursework/isiXhosa_Excellence_Sprint/Phase3/scripts) | Held-out segmentation and paired issue bootstrap |
| [Phase 4](coursework/isiXhosa_Excellence_Sprint/Phase4/scripts) | Repeated-text controls and 20 additional fits |
| [Phase 5](coursework/isiXhosa_Excellence_Sprint/Phase5/scripts) | Downstream count bigram and independent likelihood checks |
| [Phase 6](coursework/isiXhosa_Excellence_Sprint/Phase6/scripts) | Read-only scientific audit and figure-generation code |
| [One-command reproduction](scripts/reproduce_all.py) | Full original phase sequence with clean output paths and step logs |
| [Scientific source check](scripts/check_frozen_source.py) | Source fingerprints, original code and key positive/negative results |
| [Reproduction notes](docs/REPRODUCIBILITY.md) | Dataset access, environmental requirements and validity boundaries |
| [Reviewer guide](docs/REVIEWER_GUIDE.md) | Fast independent-review route |

Saved **summary** results and the training manifest are already committed.
Some large saved model binaries and per-example score tables are being imported
separately. Until that import is complete, the full saved-output audit cannot
run from a fresh GitHub clone without reconstructing the data via the original
scripts. The repository does not claim otherwise.

## Limits and Responsible NLP

- \`EnTxt\` and \`XhTxt\` are archive-assigned text views, not independently
  human-validated gold language labels.
- The main factorial experiment uses one historical newspaper. The later years
  are from that same publication, not independent source domains.
- Earlier exploratory research inspected some of the same historical years.
  The follow-up is controlled but not an untouched external validation.
- The issue-level intervals hold model seeds fixed and do not establish broad
  multilingual generalisation.
- BPC is an intrinsic count-bigram metric, not neural understanding or a
  measure of societal unfairness.
- The original newspaper transcriptions are not redistributed here because
  public distribution permissions have not been established.

## Coursework submission is separate

A GitHub repository is not the two-page extended abstract. The course also
requires the NeurIPS 2024 ethics questionnaire, the Faculty AI ethics
declaration, a contribution statement and a validated submission package.
See [Submission status](docs/SUBMISSION_STATUS.md).

## Attribution

This repository records team research. Contributor names and a code licence
must reflect actual authorship and agreed reuse rights. No licence has been
inferred or assigned.

Dataset reference: Schoots et al. (2023),
[IsiXhosa Intellectual Traditions Digital Archive](https://aclanthology.org/2023.rail-1.7/).
Tokenizer reference: Kudo and Richardson (2018),
[SentencePiece](https://aclanthology.org/D18-2012/).
Additional references are listed in the
[Phase 6 manuscript bibliography](coursework/isiXhosa_Excellence_Sprint/Phase6/CITATIONS_FOR_MANUSCRIPT.bib).
