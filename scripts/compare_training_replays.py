#!/usr/bin/env python3
"""Compare a fresh Phase 2 training run with the frozen 40-model reference.

SentencePiece can serialize the output path in its .model binary, so model
file hashes are not necessarily equal across different output directories.
We compare the actual deterministic sample hashes, learned .vocab bytes,
design and validation selection without treating a path change as a failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_rows(folder: Path) -> dict[str, dict]:
    with (folder / "model_training_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["model_name"]: row for row in rows}
    if len(rows) != 40 or len(result) != 40:
        raise AssertionError(f"Expected 40 unique tokenizer models in {folder}, got {len(rows)}")
    return result


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def compare(reference: Path, candidate: Path) -> dict:
    old, new = read_rows(reference), read_rows(candidate)
    if set(old) != set(new):
        raise AssertionError("Reference and new runs contain different tokenizer labels")
    fields = (
        "family", "arm", "seed", "actual_alpha_chars",
        "sample_hash_sha256", "vocab_sha256", "model_vocab_size",
        "validation_windows", "validation_unknown_tokens",
        "LM_training_documents", "LM_training_tokens",
    )
    mismatches = []
    changed_models = []
    for label in sorted(old):
        for field in fields:
            if old[label].get(field) != new[label].get(field):
                mismatches.append((label, field))
        old_vocab = reference / "models" / (label + ".vocab")
        new_vocab = candidate / "models" / (label + ".vocab")
        if not old_vocab.is_file() or not new_vocab.is_file():
            raise FileNotFoundError("Missing vocabulary: " + label)
        if digest(old_vocab) != digest(new_vocab):
            mismatches.append((label, "fresh_vocab_file_sha256"))
        model_path = candidate / "models" / (label + ".model")
        if not model_path.is_file() or digest(model_path) != new[label]["model_sha256"]:
            mismatches.append((label, "candidate_model_internal_sha256"))
        if old[label]["model_sha256"] != new[label]["model_sha256"]:
            changed_models.append(label)

    old_select = json.loads((reference / "PHASE2_SELECTION_FREEZE.json").read_text())
    new_select = json.loads((candidate / "PHASE2_SELECTION_FREEZE.json").read_text())
    for field in (
        "archive_sha256", "phase1_protocol_sha256", "selected_alpha",
        "models_count", "heldout_content_reads",
        "validation_windows_per_model", "candidate_grid",
    ):
        if old_select.get(field) != new_select.get(field):
            mismatches.append(("selection", field))
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "models_compared": len(old),
        "source_samples_identical": 40 - len({
            model for model, field in mismatches if field == "sample_hash_sha256"
        }),
        "learned_vocabs_identical": 40 - len({
            model for model, field in mismatches
            if field in ("vocab_sha256", "fresh_vocab_file_sha256")
        }),
        "different_serialized_model_hashes": len(changed_models),
        "selected_alpha": new_select["selected_alpha"],
        "heldout_content_reads": new_select["heldout_content_reads"],
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()
    report = compare(args.reference.resolve(), args.candidate.resolve())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit("FAIL: fresh training differs from the frozen run")


if __name__ == "__main__":
    main()
