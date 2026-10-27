# Contributing and maintaining the research record

This is a research archive and coursework implementation, not a general-purpose Python library.

1. **Create a branch and pull request.** Describe the change, the scientific files affected, and the verification run. Keep `main` reproducible.
2. **Protect the frozen study.** Avoid renaming, editing or overwriting `src/phase1/`–`src/phase6/`, the original protocols in `reports/`, saved `results/` evidence or checksummed input archive. If a scientific revision is essential, treat it as a *new version/experiment*, preserve the old analysis and document prospective vs retrospective decisions.
3. **Run checks before merging.** From the root:

   ```bash
   python -m unittest discover -s src -p 'test_*.py' -v
   python scripts/check_repository.py
   python src/run.py check
   python src/audit/independent_ci.py --root .
   ```

4. **Document provenance.** Keep source-data hashes, dependencies, seed values, sample selection, all negative findings and test reuse limitations explicit.
5. **Protect authors and archives.** Do not commit signed student declarations, private identifying details, credentials or new source archives without confirming redistribution terms. Do not invent individual contributions or silently approve draft ethics checklists.

See [repository map](docs/PROJECT_STRUCTURE.md), [reproducibility](docs/REPRODUCIBILITY.md), and [data/licensing](docs/DATA_AND_LICENSING.md).
