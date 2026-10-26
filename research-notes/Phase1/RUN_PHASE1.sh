#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
  echo "Usage: sh RUN_PHASE1.sh /path/to/TxtLangFiles1870-1894_TB.zip /path/to/isiXhosa_phase33_sampling_to_models" >&2
  exit 2
fi
SRC="$1"
PRIOR="$2"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$ROOT/outputs"
python "$ROOT/scripts/audit_phase1.py" --archive "$SRC" --phase33 "$PRIOR" --out "$ROOT/outputs" > "$ROOT/outputs/audit_log.txt"
python "$ROOT/scripts/smoke_phase1.py" --archive "$SRC" --out "$ROOT/outputs"
python "$PRIOR/scripts/verify_phase33.py" --zip "$SRC" --root "$PRIOR" > "$ROOT/outputs/legacy_phase33_reverification.json"
python "$ROOT/scripts/verify_phase1.py" --archive "$SRC" --phase33 "$PRIOR" --root "$ROOT"
