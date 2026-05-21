#!/usr/bin/env bash
# Build <team>.zip for the course "repositories" upload (excludes .venv, git, large caches).
#
# Usage: ./scripts/package_course_submission.sh team42

set -euo pipefail
TEAM="${1:?Usage: $0 <team_name> e.g. team42}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${TEAM}.zip"
rm -f "$OUT"

echo ">> Packaging ${OUT} from $(pwd)"

zip -r "$OUT" . \
  -x "./.git/*" \
  -x "./.claude/*" \
  -x "./.vscode/*" \
  -x "./.venv/*" \
  -x "./.venv_*/*" \
  -x "./outputs/*" \
  -x "./tools/__pycache__/*" \
  -x "./scripts/__pycache__/*" \
  -x "./*/__pycache__/*" \
  -x "./*.mp4" \
  -x "./WhatsApp*" \
  -x "./team*.zip" \
  -x "./submission_videos/*" \
  -x "./submission_checkpoints/*" \
  -x "./*/.DS_Store" \
  -x "./${OUT}"

echo ">> Wrote $(du -h "$OUT" | cut -f1)  $OUT"
echo ">> Upload with the repositories curl command from the course form."
