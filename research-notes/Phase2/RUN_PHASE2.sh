#!/bin/sh
set -eu
ARCHIVE="$1"
python Phase2/scripts/run_phase2.py --archive "$ARCHIVE" --output Phase2/outputs
python Phase2/scripts/verify_phase2.py --archive "$ARCHIVE" --root .
python Phase2/scripts/fresh_replay.py --archive "$ARCHIVE" --root .
