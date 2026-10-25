#!/usr/bin/env python3
"""Restore original experiment score tables and logs from verified GitHub chunks.

No historical newspaper text is distributed in this archive. The file
contains numerical outputs, validation records, experiment manifests, and
verification reports. These files support a fast audit without retraining.

Examples:
  python scripts/restore_saved_outputs.py --verify-only
  python scripts/restore_saved_outputs.py
  python scripts/restore_saved_outputs.py --destination /tmp/clean-checkout
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import os
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "SAVED_OUTPUT_CHUNKS.csv"
ARCHIVE_SHA256 = "f78f7f17893bc317d36256f68dec059679b602ae615d48d57bc569c4844fa9c2"
EXPECTED_PARTS = 97
EXPECTED_FILES = 275
ALLOWED_PREFIX = "coursework/isiXhosa_Excellence_Sprint/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_archive() -> bytes:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if len(records) != EXPECTED_PARTS:
        raise ValueError(f"Expected {EXPECTED_PARTS} chunks, found {len(records)}")
    payload = bytearray()
    for index, record in enumerate(records, start=1):
        name = f"original-coursework-outputs.tar.gz.b64.part{index:03d}"
        if int(record["part_number"]) != index or Path(record["path"]).name != name:
            raise ValueError(f"Chunk order mismatch at position {index}")
        if record["path"] != f"releases/{name}":
            raise ValueError(f"Unexpected chunk location: {record['path']}")
        encoded = ROOT / record["path"]
        try:
            raw = base64.b64decode(encoded.read_text(encoding="ascii").strip(), validate=True)
        except (OSError, UnicodeError, binascii.Error) as error:
            raise ValueError(f"Unreadable base64 payload: {encoded}") from error
        if len(raw) != int(record["chunk_size_bytes"]):
            raise ValueError(f"Chunk length mismatch: {encoded}")
        if sha256(raw) != record["chunk_sha256"]:
            raise ValueError(f"Chunk SHA256 mismatch: {encoded}")
        payload.extend(raw)
    result = bytes(payload)
    if sha256(result) != ARCHIVE_SHA256:
        raise ValueError("Whole saved-results archive SHA256 mismatch")
    return result


def safe_member(member: tarfile.TarInfo) -> bool:
    name = member.name.rstrip("/")
    parts = Path(name).parts
    if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
        return False
    if member.name.startswith("/") or ".." in parts:
        return False
    if not member.name.startswith(ALLOWED_PREFIX):
        return False
    remaining = parts[len(Path(ALLOWED_PREFIX).parts):]
    if len(remaining) < 2 or remaining[0] not in {f"Phase{i}" for i in range(1, 7)}:
        return False
    return remaining[1] == "outputs"


def verify_and_restore(*, destination: Path | None = None) -> tuple[int, int, int]:
    archive = load_archive()
    entries: list[tuple[Path, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not safe_member(member):
                raise ValueError(f"Unsafe or unexpected archive member: {member.name}")
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unreadable output file: {member.name}")
                data = extracted.read()
                if len(data) != member.size:
                    raise ValueError(f"Size mismatch: {member.name}")
                entries.append((Path(member.name), data))
    if len(entries) != EXPECTED_FILES:
        raise ValueError(f"Expected {EXPECTED_FILES} output files, found {len(entries)}")
    if len({path for path, _ in entries}) != len(entries):
        raise ValueError("Duplicated output paths in the results archive")
    if destination is None:
        return len(entries), 0, 0

    destination = destination.resolve()
    if not destination.is_dir():
        raise ValueError(f"Destination directory must exist: {destination}")
    new_files: list[tuple[Path, bytes]] = []
    unchanged = 0
    for path, data in entries:
        target = destination / path
        if target.is_symlink():
            raise ValueError(f"Refusing to follow existing symlink: {target}")
        if target.exists():
            if not target.is_file() or sha256(target.read_bytes()) != sha256(data):
                raise FileExistsError(f"Existing output differs from preserved result: {target}")
            unchanged += 1
        else:
            new_files.append((target, data))
    for target, data in new_files:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".saved-results-", delete=False) as tmp:
            tmp.write(data)
            temporary = Path(tmp.name)
        try:
            if sha256(temporary.read_bytes()) != sha256(data):
                raise ValueError(f"Temporary output verification failed: {target}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return len(entries), len(new_files), unchanged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true", help="Check archive integrity without writing files")
    p.add_argument("--destination", type=Path, default=ROOT, help="Existing destination checkout (default: repository)")
    args = p.parse_args()
    if args.verify_only and args.destination != ROOT:
        p.error("--destination cannot be used with --verify-only")
    total, restored, unchanged = verify_and_restore(destination=None if args.verify_only else args.destination)
    if args.verify_only:
        print(f"PASS: all {EXPECTED_PARTS} archive segments verified, {total} original saved output files validated")
    else:
        print(f"PASS: {total} original saved output files verified, {restored} restored, {unchanged} already identical")
        print("Next: python scripts/restore_saved_artifacts.py && python scripts/verify_saved_results.py")


if __name__ == "__main__":
    main()
