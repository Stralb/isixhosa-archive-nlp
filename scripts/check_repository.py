#!/usr/bin/env python3
"""Check that the public navigation layer resolves without altering research results."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re

ROOT = Path(__file__).resolve().parents[1]
NAV = (
    "README.md", "CONTRIBUTING.md", "docs/README.md", "docs/PROJECT_STRUCTURE.md",
    "docs/REPRODUCIBILITY.md", "docs/RESEARCH_PROGRAMME.md", "docs/DATA_AND_LICENSING.md",
    "paper/README.md", "src/README.md", "results/README.md",
    "reports/README.md", "templates/README.md",
)
REQUIRED = (
    "README.txt", "requirements.txt", "paper/report.tex", "paper/paper_references.bib",
    "paper/make_figures.py", "paper/neurips_2024_AUTHOR_REVIEW.tex",
    "src/run.py", "src/reproduce.py", "src/archivelens/audit.py",
    "reports/protocol_phase1.md", "results/segmentation.csv", "results/bigram.csv",
)

def main():
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    broken = []
    for name in NAV:
        doc = ROOT / name
        if not doc.is_file():
            missing.append(name)
            continue
        text = doc.read_text(encoding="utf-8")
        # Ignore fenced code examples. Local Markdown links only, not GitHub URLs.
        text = re.sub(r"(?ms)^\s*```.*?^\s*```", "", text)
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            parsed = urlsplit(target)
            if parsed.scheme:
                continue
            resolved = (doc.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                broken.append(f"{name} -> {target}")
    if missing or broken:
        raise SystemExit(f"Missing files: {missing}; broken local links: {broken}")
    print(f"PASS: {len(NAV)} navigation documents, {len(REQUIRED)} required paths")

if __name__ == "__main__":
    main()
