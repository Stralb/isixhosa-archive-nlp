#!/usr/bin/env python3
"""Reproduce the original six-phase coursework experiment in a clean workspace.

This runs the original frozen research scripts, It does not
rewrite those scripts, rename a source split or silently reuse saved outputs.

Usage:
  python src/reproduce.py --archive /path/to/TxtLangFiles1870-1894_TB.zip --workdir /path/to/new-run
  python scripts/reproduce_all.py --archive ... --workdir /path/to/new-run --resume
  python scripts/reproduce_all.py --archive ... --workdir /path/to/new-run --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "src"
ARCHIVE_SHA256 = "375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5"
PROTOCOL_SHA256 = "bc443d883fa1c2b8eb09ced5f87d8f65b352d55512ef2c594b1b23e64d496bf4"
STAGES = ("train", "segmentation", "repetition", "bigram", "audit")


def checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stage_commands(work: Path, source: Path, stage: str) -> list[tuple[str, list[str]]]:
    """All original phase entry points, with their original argument names."""
    py = sys.executable
    src = str(source)
    root = str(work)
    if stage == "train":
        return [
            ("phase2_train", [py, "Phase2/scripts/run_phase2.py", "--archive", src, "--output", "Phase2/outputs"]),
            ("phase2_verify", [py, "Phase2/scripts/verify_phase2.py", "--archive", src, "--root", root]),
        ]
    if stage == "segmentation":
        return [
            ("phase3_evaluate", [py, "Phase3/scripts/run_phase3.py", "--root", root, "--archive", src, "--output", "Phase3/outputs"]),
            ("phase3_verify", [py, "Phase3/scripts/verify_phase3.py", "--root", root, "--archive", src, "--output", "Phase3/outputs"]),
        ]
    if stage == "repetition":
        return [
            ("phase4_fit", [py, "Phase4/scripts/train_phase4.py", "--archive", src, "--output", "Phase4/outputs"]),
            ("phase4_evaluate", [py, "Phase4/scripts/evaluate_phase4.py", "--archive", src, "--output", "Phase4/outputs"]),
            ("phase4_verify", [py, "Phase4/scripts/verify_phase4.py", "--archive", src, "--root", root, "--output", "Phase4/outputs"]),
        ]
    if stage == "bigram":
        return [
            ("phase5_evaluate", [py, "Phase5/scripts/run_phase5.py", "--root", root, "--archive", src, "--output", "Phase5/outputs"]),
            ("phase5_verify", [py, "Phase5/scripts/verify_phase5.py", "--root", root, "--archive", src, "--output", "Phase5/outputs"]),
        ]
    if stage == "audit":
        return [
            ("phase6_audit", [py, "Phase6/scripts/audit_phase6.py", "--root", root, "--archive", src, "--out", "Phase6/outputs"]),
            ("phase6_figures", [py, "Phase6/scripts/make_phase6_figure.py", "--root", root, "--out", "Phase6/outputs"]),
        ]
    raise ValueError(stage)


def prepare_workspace(work: Path) -> None:
    if work.exists():
        if any(work.iterdir()):
            raise SystemExit("Workspace is not empty. Use --resume or choose a fresh --workdir.")
    else:
        work.mkdir(parents=True)
    for phase in range(1, 7):
        source = ORIGINAL / f"phase{phase}"
        if not source.is_dir():
            raise SystemExit(f"Missing original source phase: {source}")
        # Deliberately do not copy old output; a clean run must create its own.
        shutil.copytree(source, work / f"Phase{phase}" / "scripts")
        (work / f"Phase{phase}" / "outputs").mkdir(exist_ok=True)
        # Copy immutable preregistration files without changing their bytes.
        for name in ("FROZEN_PROTOCOL.md", "PRECOMMIT.md"):
            frozen = REPO / "reports" / {1: "protocol_phase1.md", 4: "protocol_phase4.md", 5: "protocol_phase5.md"}[phase]
            if frozen.exists():
                shutil.copy2(frozen, work / f"Phase{phase}" / name)


def execute(work: Path, label: str, args: list[str], env: dict[str, str]) -> None:
    logfile = work / "logs" / (label + ".log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("\nRUN", label, flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        with subprocess.Popen(args, cwd=work, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip("\n"), flush=True)
            rc = proc.wait()
    if rc:
        raise SystemExit(f"STOP: {label} failed with code {rc}. Inspect {logfile}")
    print("PASS", label, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Original course language ZIP")
    parser.add_argument("--workdir", type=Path, required=True, help="Clean output folder outside repository")
    parser.add_argument("--stages", default="all",
                        help="all or comma-separated stages: " + ",".join(STAGES))
    parser.add_argument("--resume", action="store_true", help="Resume from existing workspace, never delete checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without reading data or writing files")
    args = parser.parse_args()
    source = args.archive.expanduser().resolve()
    work = args.workdir.expanduser().resolve()
    selected = STAGES if args.stages == "all" else tuple(args.stages.split(","))
    if not selected or any(x not in STAGES for x in selected):
        parser.error("Invalid stages: " + args.stages)
    if work == REPO or REPO in work.parents:
        parser.error("Workdir must be outside the Git checkout, so frozen sources cannot be overwritten")
    if work == source.parent or work in source.parents:
        parser.error("Workdir cannot contain the original input archive")
    if args.dry_run:
        for stage in selected:
            for label, cmd in stage_commands(work, source, stage):
                print(label + ":", " ".join(cmd))
        return
    if not source.is_file():
        parser.error("Missing source dataset ZIP: " + str(source))
    if checksum(source) != ARCHIVE_SHA256:
        parser.error("Input ZIP hash differs from the pinned original course archive")
    proto = REPO / "reports" / "protocol_phase1.md"
    if not proto.is_file() or checksum(proto) != PROTOCOL_SHA256:
        parser.error("Frozen Phase 1 protocol missing or modified")
    if args.resume:
        for phase in range(1, 7):
            for original in (ORIGINAL / f"phase{phase}").glob("*.py"):
                copy = work / f"Phase{phase}" / "scripts" / original.name
                if not copy.is_file() or checksum(original) != checksum(copy):
                    parser.error("Cannot resume with missing or modified source: " + str(copy))
        for phase, path in ((1, "protocol_phase1.md"), (4, "protocol_phase4.md"), (5, "protocol_phase5.md")):
            dst = work / f"Phase{phase}" / ("FROZEN_PROTOCOL.md" if phase == 1 else "PRECOMMIT.md")
            if not dst.is_file() or checksum(REPO / "reports" / path) != checksum(dst):
                parser.error("Cannot resume with modified frozen protocol")
    else:
        prepare_workspace(work)
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    for stage in selected:
        for label, command in stage_commands(work, source, stage):
            execute(work, label, command, env)
    print("\nCOMPLETE", ",".join(selected), flush=True)
    print("Outputs and original script logs:", work, flush=True)


if __name__ == "__main__":
    main()
