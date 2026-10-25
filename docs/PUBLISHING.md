# Publish the complete research release

The GitHub repository already exists at https://github.com/Stralb/isixhosa-archive-nlp. It is public and contains reviewer documentation. **The full experimental source and output files are still pending import.**

1. Obtain the prepared ZIP `isixhosa-archive-nlp-public-release.zip` from the accompanying ChatGPT research handoff.
2. Review author contributions and source-data redistribution permissions. The original newspaper transcriptions were intentionally excluded.
3. Clone the existing repository and copy in the release, or use the safe existing-repository publisher in the revised ZIP. Do not create a fresh unrelated Git history or force-push.
4. Run `python scripts/check_repository.py` and `python scripts/verify_saved_results.py` after extracting the research files.
5. Push to `main` and verify that the source code, models, tables, plots, and reproducibility scripts appear online.
6. Check GitHub Actions once the full code is present and confirm that links in the README resolve.

The public repository is not the final coursework submission. The two-page extended abstract and mandatory ethics and contribution forms remain separate.
