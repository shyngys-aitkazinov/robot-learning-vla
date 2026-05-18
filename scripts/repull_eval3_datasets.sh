#!/usr/bin/env bash
# Clean re-pull of the 24 Eval 3 training-corpus datasets from HuggingFace.
#
# HF is the ground truth. This script:
#   1. Removes any locally-materialized copy under ./datasets/<name>/.
#   2. Materializes a fresh copy at ./datasets/<name>/ via
#      `huggingface_hub.snapshot_download(repo_id, local_dir=...)`. That
#      produces a self-contained tree of real files (no symlinks back into
#      the snapshot cache) so the audit / post-processor can mutate it.
#   3. Verifies the copy loads via `LeRobotDataset(root=...)`.
#
# Idempotent — re-running is cheap if the files are already up-to-date
# (snapshot_download skips files whose etag matches).
#
# Usage:
#   ./scripts/repull_eval3_datasets.sh
#   ./scripts/repull_eval3_datasets.sh --dry-run     # show targets, no actions
#   ./scripts/repull_eval3_datasets.sh --skip-clean  # don't wipe local first

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

DRY_RUN=0
SKIP_CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-clean) SKIP_CLEAN=1 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (use --help)" >&2; exit 1 ;;
  esac
done

# --- The 24 target repos (v1 + v2 raw + v2 truncated + v3) ----------------
REPOS=(
  # v1 single-celeb (3)
  "taylor_swift_1"
  "yann_lecun_1"
  "barack_obama_1"
)
# v2 raw + truncated (3 celebs x 3 positions x 2 = 18)
for celeb in taylor_swift yann_lecun barack_obama; do
  for pos in left middle right; do
    REPOS+=("dataset_v2_${celeb}_${pos}_1")
    REPOS+=("dataset_v2_${celeb}_${pos}_1_v6_truncated")
  done
done
# v3 charuco (3)
for pos in left middle right; do
  REPOS+=("dataset_v3_charuco_${pos}_1")
done

echo "Target: ${#REPOS[@]} datasets under RobotLearningVLA/"
for r in "${REPOS[@]}"; do echo "  $r"; done
if [[ "$DRY_RUN" -eq 1 ]]; then echo "(dry-run) exiting before any IO."; exit 0; fi

mkdir -p datasets
TOTAL_OK=0
TOTAL_FAIL=0

for name in "${REPOS[@]}"; do
  repo="RobotLearningVLA/$name"
  local_dir="datasets/$name"
  echo ""
  echo "================================================================"
  echo " $repo  ->  $local_dir"
  echo "================================================================"

  if [[ "$SKIP_CLEAN" -eq 0 ]] && [[ -e "$local_dir" ]]; then
    echo ">> rm -rf $local_dir"
    rm -rf "$local_dir"
  fi

  set +e
  python - "$repo" "$local_dir" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2])
local_dir.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(local_dir),
    # Skip lock files / __pycache__ etc.; we want everything else.
)

# Quick sanity: load via LeRobotDataset with the materialized root.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id, root=local_dir, video_backend="pyav")
print(f"   episodes={ds.num_episodes}  frames={ds.num_frames}  fps={ds.meta.fps}")
PY
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    echo "!! FAILED to pull $repo (rc=$rc)"
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    continue
  fi
  size=$(du -sh "$local_dir" 2>/dev/null | awk '{print $1}')
  echo ">> $local_dir  ($size)"
  TOTAL_OK=$((TOTAL_OK + 1))
done

echo ""
echo "================================================================"
echo " Summary: $TOTAL_OK ok / $TOTAL_FAIL failed (of ${#REPOS[@]} datasets)"
echo "================================================================"
total_size=$(du -sh datasets 2>/dev/null | awk '{print $1}')
echo " datasets/ total disk usage: $total_size"
echo ""
echo " Next:"
echo "   python tools/eval3_audit_dataset_labels.py"
