# Methods and reproduction

We used the class-provided IsiXIT historical language-view archive and focused on *Imvo Zabantsundu*. Sampling had two factors: page-based or word-based allocation between EnTxt and XhTxt, and page-based or word-based selection within a view. This produces PP, PW, WP and WW.

We trained on 1885 and January to September 1887. Duplicate-aware validation windows came from October to December 1887. The main held-out evaluation was 1888, with an additional within-newspaper check over 1889 to 1894.

For each policy we fitted BPE and Unigram tokenizers with a 2,000-piece vocabulary and five seeds. Each model saw about 500,000 alphabetic characters. Segmentation was evaluated in pieces per 100 alphabetic characters. The downstream count-bigram language model was evaluated in bits per alphabetic character; alpha 100 was selected from 1, 10 and 100 on validation only. Statistical intervals use paired publication-issue bootstraps with trained seeds held fixed. Twenty additional BPE models tested repeated-text and matched-support explanations.

The unmodified frozen records are [Phase 1](protocol_phase1.md), [Phase 4](protocol_phase4.md), and [Phase 5](protocol_phase5.md).

## Reproduce

Install the packages in `requirements.txt`. To check saved models and recalculate all 64 reported point effects, run:

```bash
python src/run.py check
```

For fresh training, obtain the original source archive separately. It is not included because redistribution rights have not been established. Its required SHA-256 is:

`375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`

Then run:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

The runner creates a clean Phase 1 to 6 workspace outside the repository, copies the original scripts and frozen protocols, and runs the original training and verification programs. It writes separate logs for each stage and supports `--resume`.

All 40 main tokenizers were freshly retrained in an earlier check with matching sample and learned-vocabulary hashes. Different output paths can change serialized SentencePiece `.model` hashes. **Fresh end-to-end reproduction certified on 26 October 2026.** The original pinned dataset and byte-verified GitHub source were used in a clean external workspace to regenerate the complete Phase 2–6 pipeline (11/11 stage commands, 7/7 final audit checks). All 60 SentencePiece learned models match their originals after normalizing only serialized source/output paths; the original Phase 3/4/5 held-out CSVs are byte-identical, as are the 64 final factorial contrasts. See [the complete reproduction audit](fresh_reproduction_2026-10-26.md) and [machine-readable certificate](../results/fresh_reproduction_2026-10-26.json). This was a local execution with separately supplied authorised data, not a GitHub-hosted Actions retraining job.

The archive's EnTxt and XhTxt views are not independently adjudicated gold language labels. The test years were seen during earlier exploratory work. Intrinsic tokenizer and bigram measurements do not establish societal fairness, language-wide generalisation or cultural meaning.

The repository does not replace the two-page assignment paper or its required ethics and contribution declarations.

Dataset reference: [Schoots et al. (2023)](https://aclanthology.org/2023.rail-1.7/).
