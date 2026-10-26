"""Run or verify the original historical newspaper sampling experiments."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Historical isiXhosa NLP experiments")
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("check", help="Verify saved models and report effects")
    job = commands.add_parser("reproduce", help="Fit original models using course data")
    job.add_argument("--data", required=True)
    job.add_argument("--out", required=True)
    job.add_argument("--resume", action="store_true")
    job.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    if args.cmd == "check":
        command = [sys.executable, str(src / "check.py")]
    else:
        command = [sys.executable, str(src / "reproduce.py"),
                   "--archive", args.data, "--workdir", args.out]
        if args.resume: command.append("--resume")
        if args.dry_run: command.append("--dry-run")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
