# Paper results and supporting files

The current manuscript is [`paper/report.tex`](../paper/report.tex). This page links the main findings to the data and code behind them. The older compiled paper is in [`reports/paper_draft/`](paper_draft/).

## 1. Training-data sampling and tokenisation

On *Imvo Zabantsundu* we varied two factors: how training text was allocated between English- and isiXhosa-assigned views, and how pages were selected within each view. The resulting PP, PW, WP and WW treatments were applied to BPE and Unigram tokenisers with five seeds each.

The primary 1888 results include:

- BPE segmentation, word-like versus page-like view allocation: **−1.8775** pieces per 100 alphabetic characters for the English view and **+0.4526** for the isiXhosa view.
- English-view BPE count-bigram loss: allocation effect **−0.0372 bits per character**, while the separate within-view page-selection effect was **+0.0194**.
- The 1889–1894 checks are not independent confirmation: those years were inspected during earlier experiments.

The original decisions and protocol are in [`protocol_phase1.md`](protocol_phase1.md) and [`methods.md`](methods.md). The numerical estimates are in [`results/segmentation.csv`](../results/segmentation.csv) and [`results/bigram.csv`](../results/bigram.csv). [`findings.md`](findings.md) discusses the outcomes and [`exploratory_metric_relationships_2026-10-26.md`](exploratory_metric_relationships_2026-10-26.md) examines whether segmentation and predictive loss move together.

For the initial evidence that page-based and word-based sampling yield different English-view exposure, see [the Phase 32 research decision](../archive/earlier/src/phase32/RESEARCH_DECISION.md) and [sampling estimates](../archive/earlier/results/phase32/sampling_estimands.csv).

## 2. ArchiveLens

ArchiveLens is a separate study. It uses 120 four-character sequences selected from 1887 and compares their 1885–1888 changes under four weighting rules: character mass, equal page, equal issue and equal month.

- In *Imvo*, **23/120** patterns reverse their apparent direction under an alternative rule. In *Isigidimi*, **4/120** do.
- The reversal counts were *not* unusually high in the whole-issue permutation reference. For *Imvo*, the null mean was about 29, compared with the observed 23.
- **29/120 Imvo patterns** had a Holm-adjusted difference in the *size* of the changes estimated under different weighting rules. None did in *Isigidimi*, which had much fewer issues in the studied years.
- For the example `ants`, the character-mass change was about **−0.23** and the equal-page change **+4.80** per 10,000 eligible positions. The adjusted result concerns the *gap between estimators*, not two independently established historical trends.

ArchiveLens is implemented in [`src/archivelens/audit.py`](../src/archivelens/audit.py); the reproduction commands are in [`README.txt`](../README.txt). See [initial results](ARCHIVELENS_RESULTS_2026-10-26.md), [the calibration follow-up](ARCHIVELENS_FOLLOWUP_2026-10-26.md), [claim estimates](../results/archivelens/estimands.csv), [permutation results](../results/archivelens/followup/calibration.csv) and [the original protocol](ARCHIVELENS_PRECOMMIT_2026-10-26.md).

## Reproducibility and limits

The [independent scientific audit](independent_scientific_audit_2026-10-26.md) recalculated saved predictions and statistics. A [separate fresh reproduction](fresh_reproduction_2026-10-26.md) refitted all 60 tokenisers and compared the resulting model pieces, probabilities and outputs. The GitHub Actions workflow verifies stored results; it does not retrain models. The source archive checksum and commands for checking and retraining are in [`README.txt`](../README.txt).

The available text was assigned English/isiXhosa views by the archive, not independently labelled word by word. Some evaluation excerpts repeat training-era text: the independent audit found exact 13-word overlaps in **514 of 2,493** later excerpts. The 1888 and later years had also been examined during earlier research. The four-grams are surface patterns, not annotated cultural concepts. These analyses do not establish historical opinions, editorial intent or a general social-fairness measure.

The work combines known methods in a controlled sampling study and a historical pattern-sensitivity audit; it does not propose a new tokenizer or a universal fairness metric. Relevant background is in [the related-work notes](related_work_and_archivelens_proposal_2026-10-26.md).
