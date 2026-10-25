# Reviewer guide

This document is for someone checking the claims, not someone who has followed the previous research conversations.

## Ten-minute reading order

1. Read the [main findings](FINDINGS.md) and the counterexample first.
2. Open the [Phase 1 protocol](../coursework/isiXhosa_Excellence_Sprint/Phase1/FROZEN_PROTOCOL.md).
3. Check [the Phase 6 claim ledger](../coursework/isiXhosa_Excellence_Sprint/Phase6/CLAIM_LEDGER.md), including the claims we explicitly do not make.
4. Run [the fast audit](REPRODUCIBILITY.md) with the original ZIP.
5. Read the [adversarial peer review](../coursework/isiXhosa_Excellence_Sprint/Phase6/ADVERSARIAL_PEER_REVIEW.md) and [submission gate](../coursework/isiXhosa_Excellence_Sprint/Phase6/FINAL_SUBMISSION_GATE.md).

## What to reproduce

The source fingerprint, all 40 model fingerprints, model-excerpt pairing, 32 segmentation contrasts, 32 downstream loss contrasts and four figures are all checked by the supplied audit. Full retraining is slower and is a separate reproduction route.

The raw transcript ZIP is deliberately excluded. A reviewer must obtain it independently and confirm the recorded checksum.

## Questions worth asking

- Are the English/isiXhosa categories accurately described as archive-assigned blocks rather than ground-truth labels?
- Are isolated factor contrasts being distinguished from the combined `WW` versus `PP` policy change?
- Are the negative later-year findings given equal visibility to the favourable isolated allocation result?
- Does the paper explain why lower pieces per 100 characters and lower BPC are different outcomes?
- Are the issue-bootstrap intervals interpreted conditionally, without claiming social representativeness?
- Is past exposure to evaluation years disclosed?
- Is the work appropriately identified as one historical newspaper experiment?

## Known gaps

A final course PDF, the official Moodle NeurIPS ethics form, Faculty declaration and student-authored contribution statement are not included here. The repository is not evidence that those were submitted. Refer to [Submission status](SUBMISSION_STATUS.md).