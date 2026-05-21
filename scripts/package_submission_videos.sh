#!/usr/bin/env bash
# Zip submission_videos/ for the course Azure "videos" upload.
# Usage: ./scripts/package_submission_videos.sh team42

set -euo pipefail
TEAM="${1:?Usage: $0 <team_name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VID_DIR="submission_videos"
OUT="${TEAM}-videos.zip"

if [[ ! -d "$VID_DIR" ]] || ! compgen -G "${VID_DIR}"/*.mp4 >/dev/null; then
  echo "ERROR: no ${VID_DIR}/*.mp4 — add videos first." >&2
  exit 2
fi

rm -f "$OUT"
zip -j "$OUT" "${VID_DIR}"/*.mp4
echo ">> Wrote $(du -h "$OUT" | cut -f1)  $OUT"
echo ">> Upload with the videos curl block in docs/PROJECT_SUBMISSION.md"
