# Phase 2 decision gate

- [x] Source ZIP SHA256 and Phase1 input hashes verified.
- [x] Written preregistered target, method list, scenarios, seeds, issue splits, sign threshold and uncertainty assumptions before new scores.
- [x] Clear separations between full-data oracle truth, analyst-available metadata, and hidden full-text variables.
- [x] 1885 archive split by publication issue; 80 training-ranked word forms, 68 unchanged + 12 injected.
- [x] Six selection mechanisms, whole-issue dropout, nonpositivity, repeated material, MAR and MNAR; null and injected worlds.
- [x] Held-out TEST evaluated once after development/bug-fix lock.
- [x] Unexpected NULL-as-NA CSV parsing bug documented and fixed before TEST; no scenario or scientific hypothesis changed.
- [x] Full finite-population loss and identifying-interval metrics produced, including assumption verification.
- [x] Original text recounted independently; all closed-form results and all masks numerically verified.
- [x] Fitted IPW and exact matched-cell implementations independently spot-checked.
- [x] Both DEV and TEST independent audits passed.
- [ ] Valid frequentist confidence interval calibration for selected inferences — reserved for Phase3, not claimed.
- [ ] Generalizability to wholly independent historical archive — Phase5, not claimed.
- [ ] Novel inference procedure outperforming existing baselines — NOT established.
- [ ] Human-confirmed historical semantic or cultural ground truth — NOT established.

**Decision:** Phase 2 COMPLETE as a proof-of-problem benchmark. An original algorithm is not justified yet. The next gate must compare against additional published statistical sensitivity methods on an independent evaluation design, at matched sign-decision coverage and empirical coverage targets.
