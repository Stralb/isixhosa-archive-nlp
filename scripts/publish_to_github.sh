#!/bin/sh
# Safely import the reviewed public release into an existing Git repository.
# This script preserves remote history and never force-pushes.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REMOTE=${ISIXHOSA_GIT_REMOTE:-https://github.com/Stralb/isixhosa-archive-nlp.git}

if ! command -v git >/dev/null 2>&1; then
  echo 'Git is required.' >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo 'Python is required for the offline verification scripts.' >&2
  exit 1
fi

cd "$ROOT"
python scripts/check_repository.py
python scripts/verify_saved_results.py

WORK=$(mktemp -d "${TMPDIR:-/tmp}/isixhosa-publish.XXXXXXXX")
trap 'rm -rf "$WORK"' EXIT HUP INT TERM

git clone --branch main --single-branch "$REMOTE" "$WORK/repo"
# Archive files rather than copying a local .git directory into the cloned repo.
# Any existing remote-only files remain intact.
tar -C "$ROOT" --exclude='./.git' -cf - . | tar -C "$WORK/repo" -xf -

cd "$WORK/repo"
git add -A
if git diff --cached --quiet; then
  echo 'No changes to publish. GitHub already has these files.'
  exit 0
fi
git commit -m 'Add reproducible historical isiXhosa NLP research release'
git push origin HEAD:main
printf '%s\n' 'Published the complete public release without overwriting earlier Git history.'
