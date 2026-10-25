# Reviewer guide

The central claim is that allocation and within-view selection are distinct
corpus-construction decisions. We evaluate their separate effects using four
factorial arms, two tokenizer families and five seeds, and expose the
negative whole-policy counterexample.

## A ten-minute inspection without the source archive

1. Open the actual [original training code](../coursework/isiXhosa_Excellence_Sprint/Phase2/scripts/run_phase2.py).
2. Read the [frozen Phase 1 design](../coursework/isiXhosa_Excellence_Sprint/Phase1/FROZEN_PROTOCOL.md).
3. Inspect the [original contrast tables](../coursework/isiXhosa_Excellence_Sprint/Phase3/outputs/factorial_contrasts.csv)
   and [downstream contrasts](../coursework/isiXhosa_Excellence_Sprint/Phase5/outputs/bpc_factorial_contrasts.csv).
4. Read the [claim ledger](../coursework/isiXhosa_Excellence_Sprint/Phase6/CLAIM_LEDGER.md),
   especially the claims it rejects.
5. Run \`python scripts/check_frozen_source.py\` and
   \`python -m unittest discover -s factory/tests -v\`.

These steps do not need the newspaper text and do not claim to rerun the
training. They check the public source, frozen protocols, saved summaries,
important factorial signs and the independently coded factory tests.

## Full independent replication

Obtain the original source ZIP, confirm its pinned SHA256, install the
requirements and run

\`\`\`bash
python scripts/reproduce_all.py \
  --archive /path/to/TxtLangFiles1870-1894_TB.zip \
  --workdir /path/to/a-new-folder
\`\`\`

The runner creates its own workspace and logs each training, held-out,
repetition and likelihood verification stage. Do not use the final test
years to tune hyperparameters.

## Questions worth asking

- Are the two factors actually separated by the 2 by 2 design?
- Are training exposures comparable across the four arms?
- Does the held-out improvement agree with the within-view trade-off?
- Does the full \`WW - PP\` contrast support the same story?
- Are CIs issue-clustered and conditional on tokenizer seeds?
- Are the source-view language assignments accurately described?
- Are the previous examinations of 1888 to 1894 disclosed?

The broad claim is deliberately bounded to this archive and these
experimental metrics. The separate historical-inference programme is not
an independent validation of the present factorial comparison.
