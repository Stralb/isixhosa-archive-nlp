# Historical isiXhosa NLP: sampling and measurement validity

**Research manuscript:** [*Whose Words Count? Reading Between the Samples in Historical isiXhosa NLP*](paper/README.md)  
**Dataset:** historical newspaper transcriptions from the [IsiXhosa Intellectual Traditions (IsiXIT) archive](https://aclanthology.org/2023.rail-1.7/).  
**Status:** research code and numerical evidence are available; academic submission forms still require author review.

This repository investigates how **sampling choices** change what we measure in historical isiXhosa newspaper text. It contains two related but distinct investigations:

- **Controlled tokenizer study:** a 2 × 2 comparison of between-view allocation and within-view page selection on *Imvo Zabantsundu*, using BPE/Unigram and a smoothed count-bigram model.
- **ArchiveLens:** an audit of how weighting text positions, pages, issues or months changes descriptive character-pattern comparisons between 1885 and 1888 in *Imvo* and *Isigidimi samaXosa*.

The two studies share a measurement-validity motivation; ArchiveLens is **not** a downstream effect of tokenization.

## Key results

| Study | Finding | Qualification |
| --- | --- | --- |
| Tokenizer sampling, 1888 | More word-like view allocation changes BPE segmentation by **−1.88** pieces/100 characters in the English-assigned view and **+0.45** in the isiXhosa-assigned view | A trade-off in processing archived text, not a social fairness measure |
| Downstream count-bigram | Isolated allocation changes English-view loss by **−0.0372** bits/character | Not evidence of neural or Transformer performance |
| ArchiveLens | **23/120** Imvo patterns reverse observed change direction under alternative weighting, but that count is unexceptional under whole-issue permutations | A sign reversal alone is not a verified historical trend |
| Calibrated estimator sensitivity | **29/120** Imvo patterns have Holm-adjusted *magnitude* sensitivity; **0/120** in Isigidimi | Conditional historical-archive comparison, not a linguistic or cultural discovery |

Source figures, confidence intervals, negative findings and limitations: [paper-to-evidence guide](reports/PAPER_TO_EVIDENCE_GUIDE.md).

## Quick start

Run from the **repository root** with Python 3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s src -p 'test_*.py' -v
python src/run.py check
python src/audit/independent_ci.py --root .
```

These commands **validate saved models and outputs; they do not retrain the experiment**. The same verification is checked by [GitHub Actions](.github/workflows/ci.yml).

For a clean retraining run with the verified source ZIP:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

The input file must match the hash in [data/README.md](data/README.md); the runner writes results **outside** the checkout. See [complete reproducibility instructions](docs/REPRODUCIBILITY.md).

## Where to start

| Want to... | Open |
| --- | --- |
| Read or compile the **current manuscript** | [`paper/`](paper/) |
| Understand the experiments, limitations and evidence | [`docs/`](docs/) and [paper-to-evidence guide](reports/PAPER_TO_EVIDENCE_GUIDE.md) |
| Run the original six-phase research pipeline | [`src/`](src/) |
| Use the ArchiveLens auditor | [`src/archivelens/`](src/archivelens/) |
| Inspect published figures, scores and trained models | [`results/`](results/) |
| Trace frozen protocols and historical research | [`reports/`](reports/) and [research history](reports/research_history.md) |
| Check input archive provenance | [`data/`](data/) |

The extensive `src/earlier/`, `src/historical/` and corresponding result directories are **preserved research history**, not extra required steps in the published pipeline. Their existing paths are intentionally retained for checksums, citations and reproduction.

## Responsible use and project status

- `EnTxt`/`XhTxt` are **archive-assigned text views**, not independently validated word-level language labels.
- The historical evaluation years were examined during earlier exploration. The reported controlled experiments are conditional follow-ups, not untouched independent replications.
- No surface-text result establishes historical beliefs, cultural change or societal fairness harm.
- The [NeurIPS questionnaire and author-review worksheets](paper/author_review/) are **not approved final declarations**.
- The repository currently tracks course-provided source ZIPs, but **redistribution rights still need verification**. See [data and licensing](docs/DATA_AND_LICENSING.md). No code licence has been agreed here.

**Research authors named in the manuscript:** Arno Strauss, Priyanka Gohil and Mohammed Bangie. Individual contribution roles require confirmation.

For GitHub reuse, see [contributing](CONTRIBUTING.md) and [citation metadata](CITATION.cff).
