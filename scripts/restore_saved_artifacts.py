#!/usr/bin/env python3
"""Verify or restore the exact original research binaries from portable snapshots.

Each *.b64 file is a copy of a saved original model, vocabulary or PNG encoded
as base64 so that git archives it as regular text. The SHA256 and byte count of
the decoded payload are checked against the independent manifest.

Examples:
    python scripts/restore_saved_artifacts.py --verify-only
    python scripts/restore_saved_artifacts.py
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs" / "BINARY_ARTIFACT_MANIFEST.csv"


def source_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest destination: {value}")
    if not str(relative).startswith("coursework/isiXhosa_Excellence_Sprint/Phase"):
        raise ValueError(f"Unexpected destination outside experiment outputs: {value}")
    return relative


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restore(*, verify_only: bool = False, root: Path = REPO) -> tuple[int, int]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    if len(records) != 124:
        raise ValueError(f"Expected 124 frozen model and figure artifacts, got {len(records)}")
    done, already = 0, 0
    for record in records:
        relative = source_path(record["path"])
        encoded = REPO / (str(relative) + ".b64")
        if not encoded.is_file():
            raise FileNotFoundError(f"Missing saved payload: {encoded}")
        try:
            data = base64.b64decode(encoded.read_text(encoding="ascii").strip(), validate=True)
        except (binascii.Error, UnicodeError) as exc:
            raise ValueError(f"Invalid base64 payload: {relative}") from exc
        if len(data) != int(record["size_bytes"]) or sha256(data) != record["sha256"]:
            raise ValueError(f"Original research payload does not match SHA256: {relative}")
        if verify_only:
            done += 1
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_file() and sha256(target.read_bytes()) == record["sha256"]:
                already += 1
                continue
            raise FileExistsError(f"Refusing to overwrite modified binary: {target}")
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".recovery-", delete=False) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)
        try:
            if sha256(temp_path.read_bytes()) != record["sha256"]:
                raise ValueError(f"Restore check failed: {relative}")
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)
        done += 1
    return done, already


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify all encoded payloads without writing decoded binaries")
    parser.add_argument("--output-root", type=Path, default=REPO,
                        help="Destination root for restored binaries, default current repository")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    if args.verify_only and args.output_root != REPO:
        parser.error("--output-root has no meaning with --verify-only")
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
    decoded, existing = restore(verify_only=args.verify_only, root=root)
    if args.verify_only:
        print(f"PASS: {decoded} frozen binary snapshots verified against original SHA256")
    else:
        print(f"PASS: {decoded} frozen binaries restored; {existing} identical files already present")


if __name__ == "__main__":
    main()
