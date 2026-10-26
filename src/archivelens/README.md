# ArchiveLens — historical corpus claim-sensitivity auditor

**Status:** implemented, independently checked and run on the supplied original newspaper ZIP. **Research classification:** *post-hoc conditional exploratory*; 1885 and 1888 material was examined in earlier research. The project is **not** an authentic language identifier, fairness score, sentiment model, historical-meaning classifier, or new tokenizer architecture.

## What the tool actually does

Tests how statements about historical changes in **within-word character-pattern frequencies** vary under alternative, defensible ways of weighting the *same* digitized archive. It compares the archive's `XhTxt` block views separately in Imvo Zabantsundu and Isigidimi samaXosa.

- Feature selection: top 120 within-word character 4-grams from **1887 only**, ranked by source-normalized prevalence, with no manual selection using evaluation-year outcomes.
- Target comparison: **1888 minus 1885** frequency per 10,000 within-word 4-gram positions.
- Competing sampling estimands: character-position mass (named `word_mass` in output for consistency with earlier experiments), **equal page**, **equal issue**, and **equal calendar month**.
- Uncertainty: **3,000 paired month-block/bootstrap draws** with nested resampling of whole publication issues; no page-level pseudoreplication.
- Flags: changed sign, loss of conditional bootstrap interval support, opposing supported signs subject to a magnitude floor, stable conditional support.
- Structure stress test: drop `p001` pages and see which aggregate directions reverse (a cover-page proxy, not removal of every masthead).
- Comparison of Imvo and Isigidimi is **within one digitized collection**, not independent external confirmation.
- Every reported point effect is separately reconstructed by `verify_independent.py` without importing the primary scoring/aggregation code.

## Run on the repository's verified original input ZIP

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s src/archivelens -p 'test_*.py' -v
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md
python src/archivelens/verify_independent.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
```

The input must match the SHA-256 documented in `data/README.md`. The `--verify` mode independently rebuilds the six text artifacts and compares bytes with saved evidence; the two paper figures are required to exist.

## Outputs

| File | Purpose |
| --- | --- |
| [`reports/ARCHIVELENS_RESULTS_2026-10-26.md`](../../reports/ARCHIVELENS_RESULTS_2026-10-26.md) | Exact full research report with scientific limits |
| [`reports/ARCHIVELENS_PRECOMMIT_2026-10-26.md`](../../reports/ARCHIVELENS_PRECOMMIT_2026-10-26.md) | Protocol frozen before running new ArchiveLens analysis |
| [`results/archivelens/audit.json`](../../results/archivelens/audit.json) | Machine-readable provenance, sample counts, flags and all 240 claim rows |
| [`results/archivelens/candidate_1887.csv`](../../results/archivelens/candidate_1887.csv) | Preselected patterns and 1887 source-specific rates |
| [`results/archivelens/claims.csv`](../../results/archivelens/claims.csv) | One claim per pattern and newspaper, with sensitivity classifications |
| [`results/archivelens/estimands.csv`](../../results/archivelens/estimands.csv) | 960 per-source/pattern/estimand effects and 95% intervals |
| [`results/archivelens/explorer.html`](../../results/archivelens/explorer.html) | Standalone interactive evidence explorer; download and open locally |
| [`results/figures/archivelens_claim_sensitivity.png`](../../results/figures/archivelens_claim_sensitivity.png) | Paper-ready figure (also available as vector SVG) |

The HTML explorer is self-contained: no account, internet connection or server is needed. Open it locally in a modern browser.

## Empirical result

| Descriptor | IMVO | ISIG |
| --- | ---: | ---: |
| Any alternative weighting reverses change direction | **23/120** | **4/120** |
| Word-mass baseline interval excludes zero | **38/120** | **21/120** |
| Baseline-supported effect loses support under an alternative | **9/120** | **2/120** |
| Strong supported opposite signs under two estimands | **0/120** | **0/120** |
| Removing first pages reverses baseline direction | **6/120** | **5/120** |

The **largest lesson is sensitivity, not cultural inference**. The method detects patterns whose interpretation changes or loses support when the sampling unit changes. Most sign changes concern small changes with broad uncertainty, and **none establishes a robust opposite-direction historical finding**. Indeed, two titles disagree about the sign of change for **52/120** patterns, illustrating why one newspaper should not be interpreted as all historical isiXhosa.

## Scientific validity and relation to original course experiments

This archive audit is a **separate observational study**. It must not be blended into the original held-out tokenizer F1–F6 acceptance criteria or advertised as a new confirmatory independent corpus. All text-pattern outcomes are *surface character frequencies*, not culturally adjudicated topics, genuine morphology, semantic change, or harm to isiXhosa-speaking communities.

This tool is a scoped combination of known methods—multiple estimands, grouped bootstrap, sensitivity testing and provenance—rather than proof of a first-ever auditing algorithm. Its novel coursework value is making historical **measurement stability itself an explicit, falsifiable research target** with a working reproducible tool and negative controls.
