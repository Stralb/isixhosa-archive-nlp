#!/usr/bin/env python3
"""One command-line entry point for the frozen historical isiXhosa research.

Examples:
  python scripts/factory.py audit
  python scripts/factory.py test
  python scripts/factory.py full --archive /path/to/TxtLangFiles1870-1894_TB.zip --workdir /tmp/fresh-run
  python scripts/factory.py full --archive /path/to/TxtLangFiles1870-1894_TB.zip --workdir /tmp/fresh-run --dry-run
  python scripts/factory.py smoke --archive /path/to/TxtLangFiles1870-1894_TB.zip --out /tmp/smoke-run
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(relative: str, *options: str) -> None:
    command = [sys.executable, str(ROOT / relative), *map(str, options)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def audit() -> None:
    """Verify native saved model binaries and original row-level CSV scores."""
    run("scripts/check_frozen_source.py")
    run("scripts/verify_saved_results.py")
    run("scripts/recalculate_saved_contrasts.py")
    print("PASS: 40 frozen models and 64 verified factorial point effects.")
    print("This validates saved outputs. It is not a fresh training run.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Check native model files and 64 factorial effects")
    sub.add_parser("test", help="Run independent factory unit tests")
    full = sub.add_parser("full", help="Reproduce all original six experimental phases from source")
    full.add_argument("--archive", type=Path, required=True)
    full.add_argument("--workdir", type=Path, required=True)
    full.add_argument("--dry-run", action="store_true", help="Print original experiment commands")
    full.add_argument("--resume", action="store_true", help="Resume an existing research workspace")
    smoke = sub.add_parser("smoke", help="Small developer check; not a scientific result")
    smoke.add_argument("--archive", type=Path, required=True)
    smoke.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "audit":
        audit()
    elif args.command == "test":
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT, check=True
        )
    elif args.command == "full":
        opts = ("--archive", str(args.archive), "--workdir", str(args.workdir))
        if args.dry_run:
            opts += ("--dry-run",)
        if args.resume:
            opts += ("--resume",)
        run("scripts/reproduce_all.py", *opts)
    elif args.command == "smoke":
        run("experiments/independent/run_experiment.py", "--archive", str(args.archive),
            "--out", str(args.out), "--mode", "smoke", "--bootstrap", "100")


if __name__ == "__main__":
    main()
