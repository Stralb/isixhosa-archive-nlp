# Publish the complete research release

The repository is live at https://github.com/Stralb/isixhosa-archive-nlp. Its initial reviewer documentation was committed through the connected GitHub account on 25 October 2026. The full 1,794-file research release still awaits import.

The archive `isixhosa-archive-nlp-public-release.zip` contains the coursework experiments, trained tokenizers, numerical results, code, protocol files, figures and selected supplementary research. The original historical newspaper text archives are intentionally excluded.

## Safe import into the existing GitHub repository

1. Download the prepared public-release ZIP from the ChatGPT research handoff and extract it.
2. Open a terminal inside the extracted `isixhosa-archive-nlp` folder.
3. Verify that Git is installed and authenticated to the `Stralb` account.
4. Run `sh scripts/publish_to_github.sh`.
5. The script checks the package and saved numerical claims. It clones the existing `main` branch into a temporary directory, imports the files, commits and pushes without rewriting the Git history.
6. Check that code, models, tables, plots, and the repository integrity workflow are available online.

If Git reports missing author information, configure `git config --global user.name` and `git config --global user.email`. A permission failure may mean local Git is not authenticated, even when the ChatGPT GitHub connection is authenticated.

## Limitations

The offline checks cover saved results and model fingerprints, not fresh retraining of every model. Reproduction from historical text requires the separately supplied `TxtLangFiles1870-1894_TB.zip`.

The repository does not constitute the final coursework submission. The two-page abstract and required ethics and contribution declarations remain separate.
