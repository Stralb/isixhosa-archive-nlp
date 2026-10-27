# Project structure and naming

```text
.
├── README.md              # Short public overview and runnable entry points
├── README.txt             # Course-required plain-text instructions
├── paper/                 # CURRENT manuscript source + author-review drafts
├── docs/                  # Human-readable navigation and reproducibility guides
├── src/                   # Main scientific implementation, independent checks
│   ├── phase1/ ... phase6/  # Frozen, ordered experiment workflow
│   ├── archivelens/       # Historical claim-sensitivity auditor
│   ├── analysis/          # Clearly labelled exploratory follow-ups
│   ├── audit/             # Independent saved-result checks
│   ├── earlier/           # Legacy experimental code, kept for provenance
│   └── historical/       # Separate historical-inference research programme
├── results/               # Frozen numerical evidence, models, figures and history
├── reports/               # Frozen protocols, original audits and archived paper
├── data/                  # Course-supplied archive ZIPs; rights review outstanding
├── templates/             # Original course templates
└── scripts/               # Lightweight repository-integrity checks
```

## Why not rename every `phase27` or old file?

The core `src/phase1/`–`src/phase6/` stages and their exact versions underpin published model manifests, saved hashes, and reproducibility scripts. Renaming historical experiments for appearance alone would invalidate traceability. We preserve those paths and expose **clear entry points** instead.

**Main entry point:** `python src/run.py check` (saved-output verification) or `python src/run.py reproduce --data ... --out ...` (new fitting run).

**Other valid entry points:** `src/archivelens/audit.py`, `src/archivelens/followup.py`, and `src/audit/independent_ci.py`. See [`src/README.md`](../src/README.md) for each role.

The older paper under `reports/paper_draft/` is archived by meaning, **not deleted**. The current source is `paper/report.tex`; `reports/report.tex` is a legacy scaffold and is not submission-ready.

Before a publication release, resolve missing author approvals and data redistribution terms; see [`paper/README.md`](../paper/README.md) and [data/licensing](DATA_AND_LICENSING.md).
