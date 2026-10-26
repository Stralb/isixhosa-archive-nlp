#!/usr/bin/env python3
"""Data-free check of the saved original coursework models and main effects."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"


def rows(relative):
    with (EXP / relative).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


manifest = rows("Phase2/outputs/model_training_manifest.csv")
assert len(manifest) == 40, f"Expected 40 tokenizer fits, got {len(manifest)}"
for m in manifest:
    model = EXP / "Phase2" / "outputs" / "models" / (m["model_name"] + ".model")
    assert model.exists(), f"Missing model: {model.name}"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == m["model_sha256"], f"Model hash mismatch: {model.name}"

main = rows("Phase3/outputs/factorial_contrasts.csv")
loss = rows("Phase5/outputs/bpc_factorial_contrasts.csv")
assert len(main) == 32 and len(loss) == 32


def get(items, family, period, view, contrast):
    matched = [
        r for r in items
        if (r["family"], r["period"], r["view"], r["contrast"])
        == (family, period, view, contrast)
    ]
    assert len(matched) == 1, (family, period, view, contrast)
    return float(matched[0]["mean"])


claims = [
    ("BPE English segmentation allocation", get(main, "bpe", "1888", "en", "allocation"), -1.8775366962678142),
    ("BPE isiXhosa segmentation allocation", get(main, "bpe", "1888", "xh", "allocation"), +0.4526),
    ("BPE English allocation BPC", get(loss, "bpe", "1888", "en", "allocation"), -0.03721665481863443),
    ("BPE later full policy BPC", get(loss, "bpe", "1889-94", "en", "WW_minus_PP"), +0.004503291900336492),
]
for label, actual, expected in claims:
    tolerance = 0.00006 if "isiXhosa" in label else 1e-10
    assert abs(actual - expected) <= tolerance, (label, actual, expected)

report = json.loads((EXP / "Phase6/outputs/PHASE6_SCIENCE_AUDIT.json").read_text())
assert len(report["tests"]) == 7
assert all(t["status"] == "PASS" for t in report["tests"])
print("PASS: 40 frozen models, 64 contrasts, four key findings and seven audit records")
