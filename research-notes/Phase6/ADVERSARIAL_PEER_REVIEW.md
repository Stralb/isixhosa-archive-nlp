# Adversarial peer review: scientific evidence and likely marking risks

**Verdict: RESEARCH SCIENCE AUDIT PASS; FINAL COURSEWORK SUBMISSION NOT READY.**

## Strengths supported by reproducible evidence

- Direct intervention on training-time **sampling distributions** rather than purely observational source-group comparisons.
- Full 2x2 design distinguishes language-view allocation from conditional selection within views, with 2 tokenizer families, matched corpus size, five fixed technical seeds and fixed test text.
- Truly held-out **within these specific fits**: tokenizer/LM training and duplicate-aware validation issues are dated before 1888; model selection was locked to an 1887 validation subset. Qualification: researchers had inspected future data in older projects, making prospective confirmation conditional.
- Effects have explicit point values, practical gates, whole-publication-issue bootstrap intervals, year-cluster sensitivity, negative results and reprint controls.
- Zero unknown tokens and 99,720 paired scores from original archive, independent numerical replay. Section-level source provenance and checksums are preserved.
- A negative whole-policy predictive effect in later years is disclosed rather than suppressed. This adds interpretive depth.

## Major vulnerabilities that must appear prominently

| Severity | Reviewer objection | Correct treatment |
|---|---|---|
| High | Main factorial paper concerns Imvo only, not an independent pair of newspapers | State the scope in abstract/introduction; do not call it cross-newspaper causal evidence. Historical Isigidimi analysis can motivate but is *not* the experimental test set. |
| High | Earlier phases reused future-year data | Say precommitted *follow-up*, not an untouched external confirmatory study; issue CIs cannot eliminate analyst-adaptation risk. |
| High | Small smoothed count-bigram model is an intrinsic metric | Interpret BPC as intrinsic predictive cost, not task quality, fairness to people, translation or semantic/historical understanding. |
| High | Language view not gold language label | Always write archive-assigned English-view/isiXhosa-view or `EnTxt`/`XhTxt`. Avoid ethnic identification or blanket language quality claims. |
| High | The 2024 NeurIPS questionnaire is mandatory and not in supplied ZIP | Obtain *course-specific* Moodle .tex and append completed, student-reviewed answers to final PDF, after references and before supplements. Cannot declare readiness without it. |
| High | Two-page extended abstract does not exist yet | Do not submit package as a completed coursework submission. The page count can only be validated after authorized manuscript compilation. |
| Medium | 13-word exact repeats only | Phrase F6 as a restricted robustness check on retained matched training support, not a complete duplicate-removal experiment. |
| Medium | BPE isiXhosa segmentation penalty does not map neatly to BPC | Display algorithm-dependent counterexample; avoid claiming all modelling outcomes trade off. |
| Medium | 5 seeds are *not* independent historical populations | Explain CI unit=publication issue, fixed seeds; show seed sign variability for F4. |
| Medium | Alpha 100 is grid endpoint | Describe validation optimization as selection within {1,10,100}, not an extensive hyperparameter search. |
| Medium | Source archive license has not been established | Cite creators, use original course file separately and investigate terms before public release; do not assume the paper's Creative Commons status licenses the transcribed corpus. |
| Medium | No independent genre or sentiment labels | Not fatal to this Responsibile NLP project, but limits direct historical-cultural conclusions. |
| Medium | Historical content may include offensive colonial-era language | Warn users, avoid unnecessary reproductions, respect sensitive provenance and original communities. |

## Originality challenge

Tokenization imbalance and digitized-newspaper representativeness are prior art. The distinctive claim is the *controlled* relationship among newspaper page structure, sampling policy, token representation and a bounded intrinsic downstream counterexample. Do not claim a newly invented fairness metric, new model architecture or an empirical finding about historical public opinion.

## The interpretation that earns the rubric's high Discussion band

1. The archive is not a neutral bag of text: its printed page organization and current digital view segmentation act as an input distribution for ML.
2. A page-proportional sample answers a different definition of representativeness than a word-proportional sample; neither is universally fair or scientifically canonical.
3. Changing one part of the sampling distribution can improve English-view compactness and intrinsic predictability but leave another view/algorithm worse, and within-view selection may offset the improvement.
4. Therefore dataset documentation and explicit sampling criteria are responsibilities of downstream researchers. Claims about cultural voices require additional historically grounded metadata and human expertise beyond this computational experiment.

## Decision

No new model fits recommended before the deadline. Highest marginal gain is a two-page, rigorously presented argument plus mandatory ethics materials, not searching for a larger number. An excellent rubric result is possible, but nothing in the experiment guarantees a grade.
