#!/usr/bin/env bash
# Build videos + repository zips for course submission.
# Data zip is team-specific — see submission_DATASETS.txt
#
# Usage:
#   export TEAM=team42
#   ./scripts/build_all_submission_zips.sh
#   ./scripts/build_all_submission_zips.sh --with-checkpoints

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEAM="${TEAM:?Set TEAM=teamXX first}"
WITH_CKPT=0
[[ "${1:-}" == "--with-checkpoints" ]] && WITH_CKPT=1

echo ">> TEAM=$TEAM"
./scripts/package_submission_videos.sh "$TEAM"

if [[ "$WITH_CKPT" == 1 ]]; then
  ./scripts/stage_submission_checkpoints.sh
fi

./scripts/package_course_submission.sh "$TEAM"

echo ""
echo ">> Ready for upload:"
echo "   ${TEAM}-videos.zip"
echo "   ${TEAM}.zip"
echo "   ${TEAM}-data.zip  ← build manually (see submission_DATASETS.txt)"
echo ">> Form text: SUBMISSION_SUMMARY.txt"
echo ">> Instructions: TEAMMATE_SUBMIT.md"
