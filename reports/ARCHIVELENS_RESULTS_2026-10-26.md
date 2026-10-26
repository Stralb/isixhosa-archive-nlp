# ArchiveLens: counterfactual historical-language claim stability

**26 October 2026 — post-hoc, conditional research evidence.**

## Research question

Would statements about 1885-to-1888 changes in the **surface character patterns** of historical isiXhosa archive views change under defensible alternative page-, issue-, month- or text-mass sampling estimands?

## Methods (frozen before this run)

- From the original verified source ZIP, used XhTxt block views of *Imvo Zabantsundu* and *Isigidimi samaXosa*; at least 40 lowercased ASCII words per view-page.
- Selected exactly **120 within-word character 4-grams using 1887 only**, ranking by the mean of source-normalised frequencies across both newspaper titles.
- Contrasted **1888 minus 1885** frequency per 10,000 eligible within-word four-character positions, under **word/character-mass, equal-page, equal-issue and equal-month** estimands.
- Calculated **3,000 paired calendar-month / nested publication-issue bootstrap replicates** per title, keeping all pages of an issue together. Reported percentile 95% intervals are descriptive and **not multiple-comparison-adjusted**.
- Independently recomputed point effects after excluding page 1, to probe cover-page sensitivity (not a validated masthead removal).

## Main observed results

| Newspaper | 1885 eligible pages/issues | 1888 eligible pages/issues | Change sign flips, any alternative | Word-mass supported patterns | Support lost for some alternative | Strong opposite-support conflicts |
|---|---:|---:|---:|---:|---:|---:|
| IMVO | 195/49 | 208/52 | **23/120** | 38/120 | 9/120 | 0/120 |
| ISIG | 95/12 | 96/12 | **4/120** | 21/120 | 2/120 | 0/120 |

The two newspaper titles have opposite **word-mass change directions** for **52/120** preselected patterns. This is observational disagreement between two newspaper corpora from one archive, not independent validation of a historical linguistic trajectory.

## Concrete sensitivity witnesses

The witnesses below are chosen automatically by the preregistered stability categories; **four-character sequences are not translated words or semantic concepts**.

### IMVO

| Pattern | Word-mass year change | Equal-page change | Equal-issue change | Equal-month change | Classification |
|---|---:|---:|---:|---:|---|
| `ants` | -0.23 | +4.80 | -0.03 | +0.15 | sign reversal |
| `ntsu` | -2.03 | +0.81 | -1.79 | -1.65 | sign reversal |
| `tsha` | +0.40 | -2.23 | +0.20 | +0.12 | sign reversal |
| `teto` | -0.36 | +2.21 | -0.12 | -0.29 | sign reversal |
| `iley` | +0.28 | -2.05 | +0.21 | +0.19 | sign reversal |

### ISIG

| Pattern | Word-mass year change | Equal-page change | Equal-issue change | Equal-month change | Classification |
|---|---:|---:|---:|---:|---|
| `ukub` | -0.16 | +0.49 | -0.54 | -0.54 | sign reversal |
| `aban` | -0.26 | +0.30 | +0.10 | +0.10 | sign reversal |
| `nina` | +0.00 | +0.26 | -0.07 | -0.07 | sign reversal |
| `ange` | -0.02 | +0.07 | +0.04 | +0.04 | sign reversal |
| `tsun` | -3.51 | -2.47 | -3.23 | -3.23 | support loss |

## Additional diagnostics

- **IMVO:** excluding first pages changed the word-mass direction of **6/120** candidate patterns. There were **1** excluded short views in 1885 and **0** in 1888. Common calendar months: 12/12.
- **ISIG:** excluding first pages changed the word-mass direction of **5/120** candidate patterns. There were **1** excluded short views in 1885 and **0** in 1888. Common calendar months: 12/12.

## Responsible NLP interpretation

The estimand choices describe **different exposure distributions**, not competing estimates of one uniquely correct societal fact. A conclusion that changes direction or loses conditional bootstrap support under reasonable exposure choices should be reported with the sensitivity, not elevated to a historical claim without further validation. Stability does not by itself establish historical semantic truth or social fairness.

This instrument provides a **reproducible claim-sensitivity warning**, not a language-identity detector, cultural-change detector, verified topic model, moral fairness score, or general solution to biased historical digitisation.

### Methodological limits

- The original text is corrected transcription; page views are not human-adjudicated languages or genres.
- The years 1885 and 1888 had been explored in earlier phases: this is a declared **post-hoc conditional** follow-up.
- 120 correlated string patterns are tested; bootstrap intervals are **unadjusted**, not 120 independent discoveries.
- The month/issue bootstrap measures conditional archive variability, not uncertainty about speakers, societies or missing publications.
- Source and year confound genre, editorial changes, preservation, mastheads and OCR/transcription characteristics.
- Two titles originate in the **same IsiXIT archive**; there is **no new independent modern or external validation**.

## Reproduction and artefacts

```bash
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md
python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify
```

Machine-auditable output: `results/archivelens/audit.json`, `claims.csv`, `estimands.csv`, `candidate_1887.csv`, `explorer.html`, and `results/figures/archivelens_claim_sensitivity.png` (plus vector `.svg`).

Original precommit: `reports/ARCHIVELENS_PRECOMMIT_2026-10-26.md`. Prior art/background: `reports/related_work_and_archivelens_proposal_2026-10-26.md`.
