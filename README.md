# Sampling historical isiXhosa newspaper text for NLP

We tested how the way a historical newspaper corpus is sampled changes the tokenizers trained on it. The text comes from *Imvo Zabantsundu* in the isiXhosa Intellectual Traditions (IsiXIT) collection.

The experiment changes two choices: how much text to take from the archive's English and isiXhosa views, and how to sample pages within each view. We held the training-text budget nearly constant.

## What we did

We compared four sampling policies (PP, PW, WP, WW), two SentencePiece models (BPE and Unigram), and five seeds. That produced 40 main tokenizers, each trained on roughly 500,000 alphabetic characters. We trained 20 more BPE models to check whether repeated newspaper text explained the effects.

Training used 1885 and January to September 1887. We selected smoothing on October to December 1887, evaluated on 1888, then checked 1889 to 1894.

## Main results

| Effect of changing allocation in 1888 | Difference | 95% issue-bootstrap interval |
| --- | ---: | ---: |
| English-view BPE pieces per 100 letters | -1.88 | [-1.97, -1.78] |
| English-view Unigram pieces per 100 letters | -1.79 | [-1.91, -1.67] |
| isiXhosa-view BPE pieces per 100 letters | +0.45 | [+0.41, +0.50] |
| English-view bigram bits per character | -0.0372 | [-0.0424, -0.0321] |

Lower values are better. The English-view improvement comes with a segmentation cost in the isiXhosa view. Changing the **entire** policy was not always better: it slightly worsened English-view bigram loss in the later years (+0.0045 BPC). See [findings](reports/findings.md) for the controls and limitations.

![English-view bigram results](results/figures/PHASE6_ENGLISH_LM.png)

## Run it

Install Python 3.13 and the pinned packages:

```bash
python -m pip install -r requirements.txt
python src/run.py check
```

The check verifies the saved models and recalculates all 64 factorial effects from the original paired scores. It does not retrain models.

To rerun the original experiments, obtain `TxtLangFiles1870-1894_TB.zip` from the course or the authorised archive, put it in `data/`, then run:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

Use `--dry-run` to preview the commands, or `--resume` after interruption.

## Files

| Folder | Contents |
| --- | --- |
| `src/` | Original Python experiments, runner and checks |
| `results/` | CSV comparisons, raw scores, trained models and figures |
| `reports/` | Methods, findings and frozen protocols |
| `data/` | External source data, not committed |

The original training script is in [src/phase2](src/phase2/). The key numeric tables are [segmentation](results/segmentation.csv), [bigram](results/bigram.csv) and [repetition controls](results/repetition.csv).

This study uses one newspaper. The archive's language views are not gold-standard labels, and the later years had already been inspected during earlier exploration. These results concern corpus sampling, not cultural understanding or social fairness. Details are in [methods](reports/methods.md).

Archive: [Schoots et al. (2023)](https://aclanthology.org/2023.rail-1.7/). Tokenization: [Kudo and Richardson (2018)](https://aclanthology.org/D18-2012/).
