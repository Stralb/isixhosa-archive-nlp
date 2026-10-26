# ArchiveLens frozen analysis plan (written before running the new audit)
**26 October 2026 | Research status: conditional, post-hoc with respect to known historical years.**

This protocol governs a new, self-contained tool and a new **sampling-unit sensitivity** analysis of historical isiXhosa *surface text patterns*. It does not alter the previously frozen F1-F6 tokenization experiments. The 1885/1888 excerpts and later years have been heavily studied earlier; therefore this is **not** an independently untouched test or a causal language-change study. We can still freeze tool implementation decisions before examining the new sensitivity outcomes.

## Input / unit / eligibility
- Original `TxtLangFiles1870-1894_TB.zip`, SHA-256 `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`.
- Separate newspaper collections `IMVO` (*Imvo Zabantsundu*) and `ISIG` (*Isigidimi samaXosa*). In the first instance, use `XhTxt` archive block views **only**, **not** expert language IDs or human-validated word-level labels.
- Primary date contrast **1885 versus 1888** within each newspaper. Candidate selection solely from **1887** newspaper views, never choosing target expressions based on the 1885/1888 outcomes.
- Include **all available .txt view files**, one file per newspaper issue/page/view, with at least **40 lowercased ASCII `[a-z]+` words**. Unlike earlier 90-word-window experiments, use the entire eligible corrected page-view transcription. Drop inaccessible/empty views consistently and report exclusion counts.
- Filename metadata provides source `IMVO/ISIG`, date, `p###` page number, month, and issue identifier. Cluster by source+date, not by page; 1887 feature selection is independent of the contrasted year data. No raw OCR confidence or human article-level genre is available.

## Candidate patterns, deterministic and frozen
- Count **within-word contiguous character 4-grams**, preserving multiplicity, separately from sentence/word boundaries (4-grams do **not** measure syntax).
- On all eligible 1887 `XhTxt` page views of each newspaper, compute pooled char4 counts divided by all eligible within-word 4-gram positions in that newspaper. Rank by the *equal-newspaper mean of the two normalized rates*, with lexicographic tie-breaks. Select **top K=120** patterns, without manually privileging politically suggestive strings. The candidate set is fixed before primary evaluation. Report it and its 1887 rates; avoid inferring lexical meaning from character fragments.
- Primary unit for the year difference is frequency **per 10,000 eligible within-word 4-gram positions**, `rate_1888 - rate_1885`.

## Four defensible measurement estimands
On the exact SAME eligible XhTxt page views in each source and year:
1. `word_mass`: sum pattern counts / sum all eligible 4-gram positions (a char-position-mass estimator, not a claim about full newspaper readership).
2. `page_equal`: arithmetic mean of per-page normalized rates.
3. `issue_equal`: arithmetic mean of issue-level pattern rates, where each issue pools its eligible page views by 4-gram positions.
4. `month_equal`: each of the common 12 calendar months weighted equally, using the mean issue-level rate within each month.

**Sensitivity contrasts** are differences in 1888-minus-1885 rate, and whether changing estimand alters direction or its bootstrap support. None is intrinsically the single 'correct' estimator. If a month is missing, use the intersection of months and report it rather than fill or interpolate it.

## Uncertainty, controls and decision rules
- 3,000-replicate **paired calendar-month block bootstrap**: resample the shared month labels with replacement; within every sampled month separately resample each year's publication issues with replacement, keeping each issue's page-view group intact. Use the same bootstrap draw for all candidate patterns/estimands. Do not treat pages or tokenizer seeds as independent issues.
- 95% bootstrap percentile interval per feature, source and estimand. These are **conditional descriptive intervals with no multiplicity control**; do not claim formal discovery for each of the 120 correlated patterns.
- Reference estimator is `word_mass`. A **sign reversal** means a nonzero baseline estimate and at least one of the other three estimates has opposite sign. A **supported reference** means its unadjusted 95% CI excludes zero. A **support loss** means a supported reference becomes an interval including zero under at least one other estimator. A **strong conflict** means the reference and an alternative show opposite signs, each unadjusted 95% CI excludes zero, and each absolute change is at least 2 per 10k positions. Report counts and rates with denominators, including zero results.
- Preselected structural sensitivity: recompute all four point effects after **dropping `p001` pages** from 1885 and 1888 and recalculate which reference effects reverse sign. This is a cover-page sensitivity, not a definitive masthead removal or genre control.
- Source-consistency check: report proportion of the 120 patterns with opposite **word_mass** temporal directions in IMVO vs ISIG; this is two historical newspaper environments, **not** an independent external modern/human-labeled corpus.

## Falsification / reliability / tool acceptance
- Synthetic fixture must exhibit a known page-weighted-vs-word-mass sign reversal **without changing any text**, and issue bootstrap must resample issues rather than pages.
- A synthetic fixture with proportional counts in every page must show no sampling-induced reversal.
- On real source, verify issue and month/page counts and all four aggregate rates with a second *independent formula implementation* on deterministic small subsets; verify source archive hash and write an input manifest; check result stability on a fresh rerun.
- Produce a deterministic command-line **ArchiveLens** implementation with a JSON report (machine-readable source provenance, denominators, claim categories, uncertainty); readable Markdown research report; compact CSV claim table; vector figure, and a small static HTML audit explorer if feasible. Publish only short 4-character patterns, counts/rates, and metadata, not long newspaper text.
- Run the full analysis from the *publicly committed original source ZIP* and verify all derived files with CI where feasible.
- Tool warnings must flag missing expert-validated language, human genre/topic labels, possible repetitive reprints/mastheads, archive incompleteness, potential OCR/transcription artefacts, time reuse, and different genre/source compositions. Do not invent a numerical fairness score.

## What would support a meaningful contribution?
A new and **validated audit workflow showing counterfactual claim stability** under well-specified alternative estimands, with transparent exceptions and uncertainty. Whether the observed sign-reversal rate is low, high or zero is an empirical result; do not retrofit its threshold to make the research look successful.

Any conclusion must be limited to **historical surface-form distributions conditional on this digitized corrected archive**, not historical sentiment, political agency, orthographic evolution, multilingual speaker fairness or semantic change.

## Relation to prior art
- Schoots et al. 2023, *IsiXhosa Intellectual Traditions Digital Archive*: https://aclanthology.org/2023.rail-1.7/
- Schoots 2023, root-aware political-language analysis: https://aclanthology.org/2023.rail-1.8/
- Beelen et al. 2025, `Whose News?` historical news collection/environmental scan.
- Bender & Friedman 2018, data statements; Ribeiro et al. 2020, CheckList behavioral testing.
The specific methodological components are not presented as novel inventions; the distinctive contribution is the fully reproduced archive-specific claim-sensitivity instrument, provided it works.
