#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
  echo 'Usage: sh Phase6/submission_code/RUN_FAST_CHECK.sh /path/to/TxtLangFiles1870-1894_TB.zip' >&2
  exit 2
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SOURCE=$1
cd "$ROOT"
mkdir -p Phase6/outputs
python Phase6/scripts/audit_phase6.py --root "$ROOT" --archive "$SOURCE" --out Phase6/outputs
python Phase6/scripts/make_phase6_figure.py --root "$ROOT" --out Phase6/outputs
printf '%s\n' 'FAST AUDIT AND ALL FOUR FIGURES: PASS'
