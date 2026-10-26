# Related-work audit and proposed Responsible NLP instrument — 26 October 2026

**Status: LITERATURE REVIEW AND IMPLEMENTATION PROPOSAL, NOT AN IMPLEMENTED OR VERIFIED NEW TOOL.** This document does not change the frozen Assignment 2 experiments, data, model training, F1–F6 decisions, or published outputs. The coursework deadline is **27 October 2026, 17:00 SAST**. The course explicitly prioritises conceptual depth and methodological justification; its write-up rubric assigns 30% to Discussion, 25% to Methods, and 25% to Results.

## What other researchers have actually done

| Study | Actual method / result | Why direct imitation would be weak |
|---|---|---|
| Schoots et al. (2023), *IsiXhosa Intellectual Traditions Digital Archive* | Describes provenance, context, digitisation and archive structure for historical isiXhosa newspapers/books, 1870–1914 | A simple dataset inventory is already a published contribution; we must investigate downstream methodological consequences |
| Schoots (2023), *Analyzing political formation through historical isiXhosa text analysis* | Uses isiXhosa-root-aware frequency analysis on 1874–1890 newspapers to trace emerging Black political identities | A bare `abantsundu` root counter is not novel and does not establish independent validated historical semantics |
| Ahia et al. (2023), *Do All Languages Cost the Same?* | Studies API token-price and utility disparities across 22 languages | Token-cost disparities and implications for fairness are established concepts |
| Petrov et al. (2023), *Language Model Tokenizers Introduce Unfairness Between Languages* | Compares tokenization length for parallel multilingual text, documenting wide language-dependent burdens | A token-fertility comparator alone is not a new approach |
| Ali et al. (2024), *Tokenizer Choice For LLM Training: Negligible or Crucial?* | Trains 24 LMs (~2.6B parameters) and tests whether intrinsic tokenizer metrics predict downstream performance; they often do not | Our tokenizer/BPC disagreement is a historically situated case study, not discovery of a universal first |
| Foroutan et al. (2026), *Parity-Aware Byte-Pair Encoding* | Alters BPE merges using a worst-compressed-language fair-max objective; reports inequality reductions of up to 89% | Inventing another generic fair BPE without strong new evidence would duplicate modern prior art |
| Chiu (2026), *TokLens* | Open-source tool comparing 24 tokenizers across 15 languages on six quality metrics and relating them to model benchmarks | A generic tokenizer dashboard also already exists |
| Beelen et al. (2025), *Whose news?* | Environmental Scan: enriches digitised newspapers with external reference metadata; compares proportional/equal/weighted representations and JSD | Historical representation scans are not an invented idea. Our opportunity is the **measured downstream sensitivity of specific computational conclusions** |
| Düring et al. (2024), *Transparent generosity / impresso* | Digital historical-paper interface for search, n-grams, reuse, topic, language, named entities and OCR-quality exploration | A generic newspaper explorer is not original |
| Bender & Friedman (2018), *Data Statements for NLP*; Pushkarna et al. (2022), *Data Cards* | Structured documentation of data origin, collection, purpose and limitations for responsible ML | Documentation alone is established. Build a *reproducible sensitivity check* rather than another form template |
| Ribeiro et al. (2020), *CheckList* | Behavioral testing framework with controlled test cases and perturbation-based invariance/directional expectations | Metamorphic or stress testing is not new per se; historical-source-specific tests with quantitative output are a defensible application |

### Primary bibliography

- Schoots et al. (2023), https://aclanthology.org/2023.rail-1.7/
- Schoots (2023), https://aclanthology.org/2023.rail-1.8/
- Ahia et al. (2023), https://aclanthology.org/2023.emnlp-main.614/
- Petrov et al. (2023), https://arxiv.org/abs/2305.15425
- Ali et al. (2024), https://aclanthology.org/2024.findings-naacl.247/
- Foroutan et al. (2026), https://aclanthology.org/2026.acl-long.342/
- Chiu (2026), https://aclanthology.org/2026.acl-srw.18/
- Beelen et al. (2025), https://doi.org/10.1017/chr.2025.10007
- Düring et al. (2024), https://doi.org/10.1080/01615440.2024.2344004
- Bender & Friedman (2018), https://aclanthology.org/Q18-1041/
- Pushkarna et al. (2022), https://research.google/pubs/data-cards-purposeful-and-transparent-dataset-documentation-for-responsible-ai/
- Ribeiro et al. (2020), https://aclanthology.org/2020.acl-main.442/

## Recommended tool: ArchiveLens — a Counterfactual Claim-Sensitivity Auditor

**Thesis:** Historical NLP systems often report conclusions from data whose exposure distributions depend on digitisation, source and selection policy. A computational conclusion that reverses or loses support under reasonable, disclosed alternative curation assumptions merits an explicit sensitivity warning. The tool does NOT assess whether any historical interpretation is morally fair, linguistically true or culturally authentic.

**Question:** How stable are apparent historical-language patterns and downstream NLP-model conclusions to transparent alternative choices about the unit of sampling, time/newspaper composition, text repetition and evaluation measure?

### MVP — implement using existing verified artifacts

1. **Corpus visibility cards:** newspaper/year/issue/page/view counts, archival EnTxt/XhTxt exposure under equal-page, equal-issue and equal-word weighting; explicitly distinguish the estimands.
2. **Counterfactual policy card:** for each of PP/PW/WP/WW and BPE/Unigram, compare English-view and isiXhosa-view segmentation pieces/100 alphabetic characters and count-bigram BPC; show Pareto candidates without declaring any policy ethically fair or universally best. Use publication-issue bootstrap and 5-seed variation, not page-level pseudo-independence.
3. **Sensitivity witness:** show for any selected claim its numerical estimate under multiple defensible policies, the 1888/later-year uncertainty, and whether the *direction* is stable, unstable, or inadequately supported. This must list what varied and what remained constant; never infer historical semantics from surface frequency alone.
4. **Failure-mode flags:** text repetition/13-gram overlap, language-view labels not expert gold, low lexicon hit-rate, newspaper/genre confounding, earlier exposure to held-out years, no independent external generalisation. Each flag links to a measured diagnostic; no arbitrary proprietary “fairness score.”
5. **Reproducible provenance:** emit `audit.json` / `audit.md` plus one journal-ready figure, linked back to exact source/model/result hashes, precommitted versus post-hoc status.

**Existing data supports meaningful first examples:**

- Phase 32 in matched Imvo years: English-view share is **22.643% under equal-page weighting** versus **29.475% under word weighting**; the difference is **6.832 percentage points**. These are different descriptions of the same source, not necessarily a case where one is right and the other wrong.
- Directly derived from the original **1888 BPE segmentation raw-score** totals: PP has **40.8569** pieces per 100 English-view characters and **36.5483** per 100 isiXhosa-view characters; WP has **39.0226** for English and **37.1017** for isiXhosa. The fact that better English compression comes with worse isiXhosa compression illustrates the trade-off, without establishing population-level social harm.
- The original F1–F6 validation and later complete-policy BPC counterexample show that first-factor effects do not necessarily represent all-policy changes.
- Phase 26 English-only sentiment coverage: **7.013%** of English-view tokens versus **0.276%** of isiXhosa-view tokens; **87.0%** of isiXhosa excerpts receive zero polarity. Flag "lexical resource insufficient; historic sentiment interpretation not established" rather than "historic texts were neutral." English-only TextBlob scores are *not* an isiXhosa fairness benchmark.

### Scientific extension (only if time remains after paper and declarations)

Prospectively specify a **claim stability experiment** on historical surface forms:
- Identify eligible candidate surface words/character sequences using training-only dates (avoid selecting examples after seeing changes).
- For the same 1885 versus 1888 date comparison, compute changes under equal-page vs equal-word vs equal-issue samples, and optionally month-matched/duplicate-filtered support.
- Use publication-issue bootstrap, report proportion of candidate surface changes reversing sign or losing support, and separately report amount of common-support material.
- Compare Imvo and Isigidimi **only within overlapping publication months**; do not treat Isigidimi 1870–84 versus Imvo 1889–94 as a matched historical comparison.
- For any suggested identity-related root, follow Schoots (2023) and disclose that substring and newspaper-title/masthead collisions are not human-referent labels.
- State **observed distributional drift** or **sensitivity to sampling**, NOT linguistic semantic evolution, sentiment, new cultural beliefs or fairness harm.
- Precommit rules where feasible; historical years were extensively inspected previously, so classify all new analyses as *post-hoc conditional* rather than independent tests.

### Validity guardrails

The source ZIP supplies corrected transcription **block views** and issue/page metadata; it does not supply expert word-level language identity, article genre/topic gold labels, historical semantic truth, raw OCR confidences or modern-label ground truth. Past exploration includes exact train-era 13-gram sequences in some later excerpts. Therefore do not:
- create a confident historical-language ID model from EnTxt/XhTxt without a separate human-labeled evaluation;
- infer author identity, cultural worldview, sentiment or historical political agency from surface patterns;
- advertise an original general "fair tokenization" algorithm, a new generic tokenizer quality tool, independent holdouts, or universal ethical parity;
- claim an implemented scientific tool until its code, tests and case-study outputs are actually delivered and verified.

**Rubric decision:** A demonstrably working, auditable counterfactual test with one strong controlled case study could strengthen the **25% Methods, 25% Results, and 30% Discussion** categories. A polished UI that merely re-plots F1–F6 would add little scientific value. The extended abstract, NeurIPS questionnaire, Faculty AI Ethics Statement and contribution statement remain mandatory and take precedence before the 27 October deadline.

## What would count as an original contribution?

Not a novel tokenizer architecture or discovered historic cultural shift. A defensible **historical isiXhosa case-study research instrument** that connects documented sampling estimands with same-example NLP outcome contrasts, issue-based uncertainty, temporal/source sensitivity, and automated unsupported-inference warnings. Prior methodological components have precedence; novelty is **the scoped end-to-end application and measurement to this archive**, pending empirical evidence and literature comparison.

Possible precise paper sentence **after implementation and demonstration**:

> We introduce an archive-specific counterfactual auditing workflow that exposes when a computational conclusion depends on sampling unit, publication period, or evaluation metric. On historical isiXhosa newspaper data, we demonstrate measurable language-view representation trade-offs and show that apparent model improvements are not invariant to these methodological choices.

Do NOT assert this tool exists, or attach a measured "claim reversal rate", until built and tested.
