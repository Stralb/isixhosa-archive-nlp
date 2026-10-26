Project 2 - Historical newspaper sampling and tokenizer representation
Excellence Sprint: Phases 1-5 executed; Phase 6 science audit prepared
THIS IS NOT THE FINAL COURSEWORK PDF. DO NOT SUBMIT THIS ZIP ALONE.

1. Research question

What happens to learned BPE/Unigram tokenization and smoothed count-bigram
prediction when the proportion of archive-assigned English versus isiXhosa
text, and the page-selection policy within each archive view, are separately
manipulated in historical Imvo Zabantsundu training samples?

2. Original data - REQUIRED separately

File: TxtLangFiles1870-1894_TB.zip
SHA256: 375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5
The original course-provided archive is NOT repackaged. Researchers must obtain
access through their original course materials / the IsiXhosa Intellectual
Traditions Archive. A different archive version may produce different results.
Source: Schoots et al. (2023), https://aclanthology.org/2023.rail-1.7/

3. Tested environment

Python 3.13.5, numpy==2.3.5, pandas==2.2.3, sentencepiece==0.2.1,
matplotlib==3.10.8. Install: python -m pip install -r requirements.txt
No GPU required. Tested on a CPU-backed execution environment.
If exact versions cannot install on your platform, use the environment
specified above; a newer stack has not been independently certified.

4. Fast independent verification (NO retraining)

Unzip the full package so that this README.txt is inside
isiXhosa_Excellence_Sprint/, alongside Phase1 ... Phase6.
Set SOURCE to the actual local path of the original dataset zip.

  cd /path/to/isiXhosa_Excellence_Sprint
  sh Phase6/submission_code/RUN_FAST_CHECK.sh /path/to/TxtLangFiles1870-1894_TB.zip

Expected: "PASS 7 independent scientific tests"; 40 exact trained model
hashes; 99,720 Phase3 and 99,720 Phase5 paired scores; 64 factorial point
estimates reconstructed from raw model-example outputs; four figures recreated.
See Phase6/outputs/PHASE6_SCIENCE_AUDIT.json and CLAIM_LEDGER.md.

5. Reproduce each published research phase from original archive

All paths below assume you have entered the isiXhosa_Excellence_Sprint/ root.
Use the original source ZIP path where SOURCE appears. Reproduction of full
model fitting requires more CPU time than the fast audit above.

Phase 2 (fit BPE & Unigram, choose alpha using 1887 validation only):
  python Phase2/scripts/run_phase2.py --archive SOURCE --output Phase2/outputs
  python Phase2/scripts/verify_phase2.py --archive SOURCE --root .
  python Phase2/scripts/fresh_replay.py --archive SOURCE --root .

Phase 3 (score fixed models against 1888 and 1889-1894):
  python Phase3/scripts/run_phase3.py --root . --archive SOURCE --output Phase3/outputs
  python Phase3/scripts/verify_phase3.py --root . --archive SOURCE --output Phase3/outputs
  python Phase3/scripts/make_figures.py --output Phase3/outputs

Phase 4 (reprint exclusions; train 20 additional BPE fits):
  python Phase4/scripts/train_phase4.py --archive SOURCE --output Phase4/outputs
  python Phase4/scripts/evaluate_phase4.py --archive SOURCE --output Phase4/outputs
  python Phase4/scripts/verify_phase4.py --archive SOURCE --root . --output Phase4/outputs
  python Phase4/scripts/plot_phase4.py --output Phase4/outputs

Phase 5 (count-bigram LM, alpha=100 locked from validation):
  python Phase5/scripts/run_phase5.py --root . --archive SOURCE --output Phase5/outputs
  python Phase5/scripts/verify_phase5.py --root . --archive SOURCE --output Phase5/outputs
  python Phase5/scripts/secondary_sensitivity.py --output Phase5/outputs
  python Phase5/scripts/plot_phase5.py --output Phase5/outputs

For a truly clean replay use a COPY of the whole research directory and
remove the generated output files from the copy first. Never mutate the
frozen Phase1 protocol or Phase2 selection lock. Phase4 and Phase5 have
restart-safe checkpoints. Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 to
reduce thread contention where appropriate.

Phase 1's historical audit can be replayed with its RUN_PHASE1.sh script,
but also requires the earlier (outside this Excellence Sprint) Phase33
reference directory preserved in dependencies/Phase33_baseline.zip.
See Phase1/START_HERE.md for the exact path and invocation. The fast Phase6
check does not need the earlier Phase33 directory.

6. Method synopsis

Training: Imvo 1885 + Jan-Sep 1887, same 500,000 alphabetic letters per
factorial tokenizer; 2 families x 4 policies x 5 seeds. Validation: 1887
Oct-Dec duplicate-aware text, global smoothing alpha chosen from {1,10,100}.
Test: 1888, followed by later Imvo 1889-94. Evaluation is paired across
40 models for each archive-assigned page/view. Publication issue bootstraps
provide CIs; they do not confer independence across previous research phases.

7. Where to find results

Phase3/outputs/factorial_contrasts.csv: 32 segmentation contrasts.
Phase4/outputs/f6_contrasts.csv: repeated-text robustness.
Phase5/outputs/bpc_factorial_contrasts.csv: 32 bigram-loss contrasts.
Phase6/outputs/PHASE6_SCIENCE_AUDIT.json: independent scientific audit.
Phase6/CLAIM_LEDGER.md: reportable findings and false claims to avoid.
Phase6/FINAL_SUBMISSION_GATE.md: mandatory coursework blockers.
Phase6/outputs/*.png: figures rendered from verified results.
Phase6/CITATIONS_FOR_MANUSCRIPT.bib: reviewed scholarly references.

8. Essential scope/ethics

The main Excellence Sprint investigates one historical newspaper (Imvo),
not all historic isiXhosa writing. EnTxt/XhTxt are archive block categories,
not human-validated language labels. Smaller token cost and bigram BPC do NOT
prove social fairness, meaningful downstream NLP performance or changes in
historical culture. Prior phases inspected the held-out years; verification
of code must not be called untouched external validation. Verify permissions
before public redistribution of corpus text. See Phase6 ethical worksheet.

9. Coursework submission requirements - STILL UNFINISHED

The final 2-content-page extended abstract PDF is NOT inside this code ZIP.
The Moodle-specific 2024 NeurIPS ethics .tex, official Faculty of Science
AI declaration, real human contribution statement and final PDF verification
must be completed by students. The assignment says no mark is given without
the NeurIPS questionnaire. Deadline: 27 Oct 2026 17:00 SAST.
