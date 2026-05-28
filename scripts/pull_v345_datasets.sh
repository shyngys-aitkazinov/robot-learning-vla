#!/usr/bin/env bash
# Pull every RobotLearningVLA/dataset_v{3,4,5}_* dataset into ./datasets/<name>/.
# Uses snapshot_download (full real files, no symlinks) so the audit/post-process
# tools can mutate the trees. Idempotent: snapshot_download skips unchanged files.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO_LIST="${1:-/tmp/v345_repos.txt}"
mkdir -p datasets
LOG_DIR="$ROOT/datasets/.pull_logs"
mkdir -p "$LOG_DIR"

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL=0

# Skip the count header (first line) if present.
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  case "$name" in
    dataset_v3_*|dataset_v4_*|dataset_v5_*) ;;
    *) continue ;;
  esac
  TOTAL=$((TOTAL + 1))
  repo="RobotLearningVLA/$name"
  local_dir="datasets/$name"
  log="$LOG_DIR/$name.log"
  echo ""
  echo "[$TOTAL] $repo  ->  $local_dir"

  python - "$repo" "$local_dir" >"$log" 2>&1 <<'PY'
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
)
print("snapshot_download OK")
PY
  rc=$?

  if [[ "$rc" -ne 0 ]]; then
    echo "  !! FAILED rc=$rc (see $log)"
    tail -n 8 "$log" | sed 's/^/     | /'
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    continue
  fi
  size=$(du -sh "$local_dir" 2>/dev/null | awk '{print $1}')
  echo "  ok ($size)"
  TOTAL_OK=$((TOTAL_OK + 1))
done < "$REPO_LIST"

echo ""
echo "=============================================================="
echo " Summary: $TOTAL_OK ok / $TOTAL_FAIL failed (of $TOTAL targets)"
echo "=============================================================="
total_size=$(du -sh datasets 2>/dev/null | awk '{print $1}')
echo " datasets/ total disk usage: $total_size"
