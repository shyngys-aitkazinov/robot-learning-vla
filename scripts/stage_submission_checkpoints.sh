#!/usr/bin/env bash
# Copy Hub policy snapshots into submission_checkpoints/ for course repo zip.
# Run once before package_course_submission.sh if graders may not use HF.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

DEST="submission_checkpoints"
mkdir -p "$DEST"

stage() {
  local name="$1" repo="$2"
  local out="${DEST}/${name}"
  echo ">> Downloading ${repo} → ${out}"
  python3 - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
repo = "${repo}"
out = Path("${out}")
out.mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo, repo_type="model")
src = Path(path)
for f in src.iterdir():
    dst = out / f.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if f.is_symlink():
        dst.symlink_to(f.resolve())
    else:
        shutil.copy2(f, dst)
print("Staged", out)
PY
}

stage "v4slots_expert_50k" "RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k"
stage "v16_pinsv5_step5k" "RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k"

echo ">> Done. Checkpoints under ${DEST}/"
