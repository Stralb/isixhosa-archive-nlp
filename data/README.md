# Original isiXhosa newspaper source archives

The two original course-provided ZIPs belong in this folder. Keep them **compressed**: do not extract the text into Git or rename files *inside* either ZIP.

| Repository path | SHA-256 | Archive purpose |
| --- | --- | --- |
| `data/TxtLangFiles1870-1894_TB.zip` | `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5` | Source file required by the six-phase experiment |
| `data/TxtFiles1870-1894_TB.zip` | `41fc2a7cab043025df3e2127ee21878c5ae5b6457a8bd55b5e210fb1c50cde10` | Original combined-view newspaper text archive; supplied under the filename `1. TxtFiles1870-1894_TB.zip` |

Both original local ZIP files were checked with `sha256sum` and opened successfully on 26 October 2026 (14,442 and 7,220 ZIP members respectively). These names are not interchangeable.

To verify after copying the files into the repository:

```bash
sha256sum data/TxtLangFiles1870-1894_TB.zip data/TxtFiles1870-1894_TB.zip
python src/run.py check
```

To do a fresh reproduction from the language-view ZIP:

```bash
python src/run.py reproduce --data data/TxtLangFiles1870-1894_TB.zip --out ../isixhosa-replay
```

The dataset is the course-supplied historical material; including it in GitHub is an explicit repository-distribution decision by the project maintainers. Attribution and any archival redistribution conditions must still be respected. The raw ZIP is approximately 29 MB and the combined-view ZIP approximately 26 MB; ordinary Git can track both individually (each is below GitHub's 50 MB warning threshold).
