# Assignment 2 rubric and two-page evidence blueprint (not a manuscript)

Source: the original Wits COMS4054A/COMS7062A/COMS7066A 2026 Project 2 brief, pp. 2-5.

| Rubric | Weight | Evidence that is actually available | Marking risk to avoid |
|---|---:|---|---|
| Structure and Introduction | 10% | One precise historical-data Responsible NLP question, conceptual pipeline and significance | A list of 33 experiments, formatting problems, or >2 main-body pages. **Negative marking** indicated. |
| Background and Related Work | 10% | Schoots, tokenizer research, and historical newspaper corpus-bias research | Pretending the general tokenizer effect is novel; missing citations or irrelevant literature. **Negative marking** indicated. |
| Method | 25% | 2x2 paired sampling experiment, constant 500k letters/model, 2 families, 5 seeds, 1887 validation lock, issue CIs, repeat diagnostic | Unclear cause/measurement separation, claiming no leakage despite research-program test reuse, or overclaiming broad hyperparameter optimization. |
| Results | 25% | English allocation -1.878 pieces/100 chars and -0.0372 BPC in 1888; both later-year negative; isiXhosa segmentation trade-off; repeat robustness; full-policy negative result | Reporting only positive tests, lacking calibrated baselines and confidence intervals, unreadable figures or not specifying units. |
| Discussion | **30%** | Archive layout makes sampling representationally consequential; page vs word policies target competing notions of balance; improved tokens need not imply model-level or societal fairness; historical agency cannot be inferred from archive-view tags | Mistaking tool metrics for social harm/cultural change or omitting the negative downstream result. |

## Main body page-budget map (for paper author to use LATER)

- **Content page 1:** a focused motivation and scholarly positioning, explicit research question, compact factorial method schematic and true train/validation/test chronology. Explain the data and 2x2 treatment rather than enumerating phases.
- **Content page 2:** one legible English segmentation/BPC figure plus compact numerical outcomes including isiXhosa tradeoff; reserve substantial area for Discussion, counterexample, limitations and responsible practical implications.
- **After page 2, outside main content limit:** references and verified contribution statement (unlimited according to course brief), mandatory NeurIPS 2024 Moodle ethics statement *after references and before supplementary*, Faculty Science AI ethics statement, any supplementary material. Validate exact order against local course template.

These are editorial recommendations, **not written abstract paragraphs**. The author's final two-page PDF must be compiled and visually checked in a later explicitly authorized activity.

## Marking facts

The lecturer values methodological reasoning and appropriate metrics above maximizing model scores, rewards informative failures, permits writing/coding assistance with explicit responsibility, and penalizes hallucinations and poor formatting. The required deliverables are code, `requirements.txt`, `README.txt`, and a PDF extended abstract. Submission due **27 October 2026, 17:00 Africa/Johannesburg**.
