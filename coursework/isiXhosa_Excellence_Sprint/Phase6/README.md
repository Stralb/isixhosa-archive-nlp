# Excellence Sprint Phase 6 - final scientific peer review and submission audit

**Phase status:** Scientific review, numerical verification, package preparation and grader-facing code documentation DONE; **COURSEWORK SUBMISSION STILL BLOCKED** until the explicitly authorized two-page PDF and actual course ethics/contribution forms have been completed. This folder contains *no fabricated PDF or filled ethics form*.

## What Phase 6 actually did

1. Re-read and visually inspected the original 2026 Project 2 brief, including the 10/10/25/25/30 rubric and required Moodle questionnaire.
2. Independently verified source ZIP hash, frozen Phase 1 protocol hash, Phase 2 validation lock, all 40 tokenizer/vocab hashes, matching 99,720 Phase 3 versus Phase 5 per-example token counts, and recomputed **all 64 factorial point effects** without fitting a model.
3. Confirmed the main BPE English-view 1888 F5 effect **-0.03721665 BPC** [-0.04236444,-0.03211884] (publication-issue confidence interval), about 1.29% of PP baseline, and reproduced the later **+0.0045033 BPC** WW-minus-PP counterexample. Also audited the reprint-control and bilingual tradeoff logic.
4. Prepared clean, source-derived effect illustrations (segmentation, downstream allocation, full-policy counterexample, isiXhosa effect), a scientific claim ledger, adversarial review, rubric map, bibliography, user-facing verification script, pinned environment and a source-data nonredistribution policy.
5. Produced a *human-review worksheet*, NOT the required NeurIPS or Faculty paperwork. Both forms remain to be supplied/completed by students. The written extended abstract remains intentionally absent.

## Quick check

```
sh Phase6/submission_code/RUN_FAST_CHECK.sh /path/to/TxtLangFiles1870-1894_TB.zip
```

Requires Python 3.13.5 and the pinned `requirements.txt` at project root. On successful run the terminal prints `PASS 7 independent scientific tests` and `FAST AUDIT AND ALL FOUR FIGURES: PASS`.

## Documents

- `CLAIM_LEDGER.md`: tested claims, exact numbers, counterexamples and forbidden overclaims.
- `ADVERSARIAL_PEER_REVIEW.md`: critical red-team objections, countermeasures and originality positioning.
- `RUBRIC_AND_PAGE_BUDGET.md`: exact grading categories and suggested evidence placement, **not a paper draft**.
- `ETHICS_AND_CONTRIBUTION_WORKSHEET.md`: prompts requiring real human answers and genuine Moodle form.
- `FINAL_SUBMISSION_GATE.md`: mandatory gates, science/pass versus submission/block distinctions.
- `CITATIONS_FOR_MANUSCRIPT.bib`: verified bibliographic candidates.
- `scripts/audit_phase6.py`, `scripts/make_phase6_figure.py`: reproducible code, no new model fitting.
- `outputs/PHASE6_SCIENCE_AUDIT.json`: machine-readable independent review.
- `outputs/PHASE6_VERIFIED_BPC_CONTRASTS.csv`: factor contrasts recomputed from raw saved scores.
- `outputs/PHASE6_*png`: four illustrative plots from frozen estimates and 95% issue-level intervals.
- Root `README.txt` and `requirements.txt`: correct names for grader.

## Important limitations

Phase 6 does not certify that graders will award any particular mark. The later tests all use issues from Imvo, not an unrelated corpus, and those years were visited in preceding analyses. Archive language-view tags are not independently gold-labelled. BPC is a small count-model's intrinsic measure, not a user-facing fairness score. Reprint exclusion is exact-string and changes support. Never claim external independence or modern isiXhosa task accuracy.
