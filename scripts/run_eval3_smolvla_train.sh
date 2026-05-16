#!/usr/bin/env bash
# Fine-tune SmolVLA for Eval 3 on single-camera Hub data (e.g. observation.images.front).
# lerobot/smolvla_base expects observation.images.camera{1,2,3}; we map front→camera1 and pad 2 views.
#
# Prereqs:
#   uv pip install transformers accelerate sentencepiece
#
# Usage:
#   ./scripts/run_eval3_smolvla_train.sh
#   EVAL3_TRAIN_STEPS=2000 EVAL3_BATCH=4 ./scripts/run_eval3_smolvla_train.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="${EVAL3_DATASET_REPO:-RobotLearningVLA/taylor_swift_1}"
OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_smolvla}"
JOB="${EVAL3_JOB_NAME:-eval3_smolvla}"
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
BATCH="${EVAL3_BATCH:-8}"
DEVICE="${EVAL3_POLICY_DEVICE:-mps}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=2 \
  --rename_map="$RENAMES" \
  --dataset.repo_id="$REPO" \
  --dataset.video_backend=pyav \
  --job_name="$JOB" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  "$@"
