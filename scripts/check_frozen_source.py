#!/usr/bin/env python3
"""Source-only, data-free checks for the frozen project and public claims."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT

def locate(rel):
    if "/outputs/" in rel:
        return EXP / "results" / rel
    if rel.endswith("FROZEN_PROTOCOL.md") or rel.endswith("PRECOMMIT.md"):
        return EXP / "protocols" / rel
    return EXP / "experiments" / rel
KNOWN = {
    "Phase1/FROZEN_PROTOCOL.md": "bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4",
    "Phase4/PRECOMMIT.md": "1fb49e638a51cedc7f8157743d2c2d90ba65003f4d049990cf8d23c30f00714e",
    "Phase5/PRECOMMIT.md": "8575dc1ae52a22391318933bc5d4c8cee8777c15645a9eed13bbe34bb15583f8",
}
for rel, expected in KNOWN.items():
    path = locate(rel)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssertionError("Original frozen artifact differs: " + rel)

source_files = list((EXP / "experiments").glob("Phase[1-6]/scripts/*.py"))
if len(source_files) < 18:
    raise AssertionError(f"Expected actual original Python scripts; found {len(source_files)}")
for path in source_files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

def rows(rel):
    with (locate(rel)).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

models = rows("Phase2/outputs/model_training_manifest.csv")
segmentation = rows("Phase3/outputs/factorial_contrasts.csv")
lm = rows("Phase5/outputs/bpc_factorial_contrasts.csv")
repeat = rows("Phase4/outputs/f6_contrasts.csv")
if len(models) != 40 or len(segmentation) != 32 or len(lm) != 32 or not repeat:
    raise AssertionError("The expected main saved manifests and contrasts are incomplete")

found = {(x["family"], x["period"], x["view"], x["contrast"]): x for x in lm}
primary = found[("bpe", "1888", "en", "allocation")]
later = found[("bpe", "1889-94", "en", "WW_minus_PP")]
if not (float(primary["mean"]) < 0 and float(primary["ci_high"]) < 0 and float(later["mean"]) > 0):
    raise AssertionError("Published findings or counterexample unexpectedly changed")

for candidate in ("TxtLangFiles1870-1894_TB.zip", "1. TxtFiles1870-1894_TB.zip"):
    if list(ROOT.rglob(candidate)):
        raise AssertionError("Original archival dataset should not be redistributed")

print("PASS: exact source hashes, original code syntax, 40 models in manifest,")
print("      32 segmentation + 32 LM contrasts, robustness and negative counterexample")
print("NOTE: Source-only checks are not full raw-data reproduction or model re-training.")
