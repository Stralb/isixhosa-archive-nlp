# Human-review worksheet for required 2026 coursework statements — NOT completed answers

**STOP: THIS FILE IS NOT THE ACTUAL NEURIPS QUESTIONNAIRE AND CANNOT REPLACE IT.** The course requires the exact *2024 NeurIPS ethics statement/questionnaire .tex supplied on Moodle*, appended after references and before supplementary material. That file is absent from the original user-supplied extended abstract template ZIP. The NeurIPS official checklist guidance (https://neurips.cc/public/guides/PaperChecklist) is a helpful source on responsible answering, but *the course's own Moodle file is the submission authority*. Do not silently substitute a newer conference template. Human authors must decide and sign as required.

## Checklist topics: fill the Moodle .tex, question by question

| Topic | What the existing research can substantiate | Human decision still necessary |
|---|---|---|
| Claims | Archive language-view sampling and count-bigram effect, strictly bounded | Confirm manuscript claims mirror `CLAIM_LEDGER.md` |
| Limitations | One newspaper, reused test years, labels not gold, no human impact, modest BPC | Ensure disclosed in final paper |
| Theory | No theoretical theorem or proof was offered | Whether course form permits N/A |
| Experimental reproducibility | 40 models, original source hash, code and 99,720 score replay | Confirm code and original source are accessible to marker |
| Open code/data | Code released to lecturer; original newspaper ZIP supplied separately | Confirm public release permissions/availability, do not assume |
| Experimental detail | Train 1885 & 1887 Jan-Sep; val 1887 Q4; test 1888 and 1889-94; fixed alpha grid | Confirm final report includes essentials |
| Statistical uncertainty | Issue-level paired 5,000-draw percentile bootstrap + later 6-year cluster diagnostic | Confirm appropriate notation, interpretation |
| Compute resources | CPU-based Python and SentencePiece execution, small LM | Record real CPU/memory/run time from own machine if required; don't invent |
| Ethics/consent | Historical public newspapers handled as archived cultural heritage | Review university-specific research ethics policies; do not claim board clearance |
| Broader impact | Transparent corpus curation can reduce unjustified generalizations; careless release could erase context or reinforce colonial framings | Human researcher assesses risks and mitigations |
| Safeguards | This work does not release a powerful generative model | Confirm original corpus license and release scope |
| Existing-asset licenses | Schoots et al. archive citation, SentencePiece Apache-2.0 library; papers properly cited | Verify **actual corpus text license** separately; article license != data license |
| New assets | Reproducibility scripts/tokenizers and derived numerical outputs | Confirm what may be released and under which terms |
| Human participants | No contemporary annotator or human-subject experiment was performed in these phases | Verify other group members' work does not change this |
| Institutional review | No approval requested or claimed in these files | Determine applicable policy; do not manufacture IRB/REC approval |
| LLM usage | Substantial assistant help was used in scientific planning, implementation, analysis, QA and code/text | Disclose truthfully under the Faculty AI statement and any applicable course requirements |

These are **prompts, not suggested Yes/No selections**. The numbered and worded sections of the actual Moodle questionnaire take precedence.

## Preliminary contribution statement and final contribution statement

The Project 2 brief requires a preliminary contribution statement **two weeks before** the submission. The chat/files do not establish whether this requirement was met. Human students must confirm original group membership and any preliminary submission; do not backdate.

- Real student name/number and group membership: `[STUDENTS TO COMPLETE]`.
- Specific ownership of design, coding, verification, literature, writing and experiment interpretation: `[STUDENTS TO ALLOCATE TRUTHFULLY]`.
- Contribution of AI assistants to methodology, code, exploratory analyses, scientific interpretation and any future drafting: `[STUDENTS TO DISCLOSE]`.
- Independent human review of numerical claims, code plagiarism/attribution and final manuscript: `[STUDENTS TO VERIFY]`.
- Attribution and licence checks for Schoots dataset, article, tokenizer implementation and other assets: `[STUDENTS TO VERIFY]`.

## Faculty of Science AI Ethics Statement

The correct Faculty wording/format must be sourced from the course resources. The supplied archive does not contain a fillable official Faculty form. Do not state that an unverified generic paragraph fulfils the official declaration. The statement should accurately disclose AI-assisted experiment planning and code generation, not merely proofreading, if that matches actual use.

## Ethical limitations worth discussing, independent of questionnaire

Historic newspapers include content from a colonial political setting, and the archive is a curated set of transcribed/edited materials. The corpus's language-view segmentation is not a benchmark of native-speaker competence. Fairness to present-day isiXhosa users, cultural interpretation and harm assessment require stakeholder input and validated human tasks. Avoid attributing beliefs to individual authors or speakers based on training-data proportions alone.
