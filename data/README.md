# Newspaper archive files

These are the original course-supplied newspaper transcriptions. Keep the ZIP contents unchanged: the experiments refer to the archive's original filenames and structure.

- `TxtLangFiles1870-1894_TB.zip` — the language-view archive used by the experiments. SHA-256: `375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5`.
- `TxtFiles1870-1894_TB.zip` — the combined-view archive. SHA-256: `41fc2a7cab043025df3e2127ee21878c5ae5b6457a8bd55b5e210fb1c50cde10`.

Check the files before running the experiments:

```bash
sha256sum data/TxtLangFiles1870-1894_TB.zip data/TxtFiles1870-1894_TB.zip
```

The data came from the course materials associated with the [IsiXIT newspaper corpus](https://aclanthology.org/2023.rail-1.7/).

**Redistribution permission has not yet been verified.** Both ZIPs are currently tracked in this public repository, but being available on GitHub does not establish a licence to redistribute them. The group should confirm the source terms with the relevant archive or course owners. Removing the files in a later commit would not remove them from Git history.

The `EnTxt` and `XhTxt` blocks are labels supplied by the archive, not independently checked word-level language annotations.
