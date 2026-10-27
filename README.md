# Historical isiXhosa newspaper NLP

This is our NLP Project 2 at the University of the Witwatersrand. We work with digitised nineteenth-century editions of *Imvo Zabantsundu* and *Isigidimi sama-Xosa* to ask a fairly practical question: how much do the results change when we sample and count newspaper text in different ways?

We ran two related studies:

- **Tokenisation:** On *Imvo*, we varied how training text was allocated between the archive's English and isiXhosa views, and how pages were sampled within each view. We compared BPE and Unigram tokenisers, then evaluated a simple count-bigram language model.
- **ArchiveLens:** We compared changes in 120 character four-grams between 1885 and 1888, weighting the text by character positions, pages, issues or months. We also checked how sensitive the findings were to issue sampling and repeated passages.

These are separate experiments. ArchiveLens does not use the tokeniser outputs.

## Findings

For the 1888 test set, word-based allocation reduced English-view BPE segmentation by about **1.88 pieces per 100 characters**, but increased the isiXhosa-view cost by **0.45**. English-view bigram loss fell by **0.037 bits per character** under the same allocation contrast.

In ArchiveLens, 23 of 120 *Imvo* patterns changed the direction of their apparent 1885–1888 trend under different weighting rules. That number was not unusual in the permutation test, although 29 patterns showed significant differences in the *size* of their estimates after Holm correction.

The results concern the available digitised text, not historical opinions or a measure of social fairness. The archive's language labels were not checked word by word. The [paper](paper/README.md) and [supporting results](reports/PAPER_TO_EVIDENCE_GUIDE.md) explain the methods and limitations in detail.

## Running the project

Use Python 3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m unittest discover -s src -p 'test_*.py' -v
python src/run.py check
```

This checks the saved experiment outputs; it does **not** retrain the models.

To rerun training from the original source archive:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

The full run is more expensive. The archive ZIP must match the checksum in [`data/README.md`](data/README.md). Additional verification commands, including ArchiveLens, are in [`README.txt`](README.txt), which accompanies the coursework submission.

## Files

- [`paper/`](paper/) — latest paper source, figures and build instructions.
- [`src/`](src/) — experimental code; `src/run.py` is the main entry point.
- [`results/`](results/) — recorded model outputs, scores and figures.
- [`reports/`](reports/) — methods, results, protocols and research checks.
- [`data/`](data/) — source archives and their checksums.

Earlier exploratory experiments are retained under `src/earlier/`, `src/historical/` and their matching results directories.

**Authors:** Arno Strauss, Priyanka Gohil and Mohammed Bangie. Individual contributions are to be confirmed in the course submission.
