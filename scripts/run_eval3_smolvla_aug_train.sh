#!/usr/bin/env bash
# Eval3 SmolVLA fine-tune wrapper WITH the full data-prep + augmentation stack:
#   * Layer 1 (preprocessing): per-episode frame truncation to 600 frames (= 20s @ 30fps)
#     so the model only trains on 20s-budget-fittable trajectories. Driven by
#     env var EVAL3_MAX_FRAMES_PER_EP (default 600). 17/18 Obama episodes exceed 20s.
#   * Layer 2 (preprocessing): random task-string augmentation at load time.
#     Driven by EVAL3_TASK_AUG (default 1). The recordings carry "Place the coke on
#     the <X>" but demo prompts will be "Place the coke on <X>". This layer mixes
#     6 prompt variants weighted 40% canonical demo wording.
#   * Layer 3 (augmentation): torchvision image transforms — brightness/contrast/
#     saturation/hue/sharpness/affine. Brightness + contrast are weighted 2x because
#     they're the strongest defence against the LeCun↔Obama spurious lighting cue
#     (LeCun luma 0.660, Obama luma 0.389 — 0.27 gap, almost 2x the Swift↔LeCun gap).
#
# Required deps (installed once):
#   uv pip install transformers accelerate sentencepiece num2words
#
# Usage:
#   ./scripts/run_eval3_smolvla_aug_train.sh
#   EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=1 ./scripts/run_eval3_smolvla_aug_train.sh   # MPS smoke
#   EVAL3_POLICY_DEVICE=cuda EVAL3_TRAIN_STEPS=50000 EVAL3_BATCH=8 \
#     EVAL3_EXTRA_REPOS=RobotLearningVLA/yann_lecun_1,RobotLearningVLA/barack_obama_1 \
#     ./scripts/run_eval3_smolvla_aug_train.sh                                     # Brev

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="${EVAL3_DATASET_REPO:-RobotLearningVLA/taylor_swift_1}"
OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_smolvla_aug}"
JOB="${EVAL3_JOB_NAME:-eval3_smolvla_aug}"
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
BATCH="${EVAL3_BATCH:-8}"
DEVICE="${EVAL3_POLICY_DEVICE:-mps}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'

# Layer 1+2 prep is on by default; surface knobs for ablation runs.
export EVAL3_MAX_FRAMES_PER_EP="${EVAL3_MAX_FRAMES_PER_EP:-600}"
export EVAL3_TASK_AUG="${EVAL3_TASK_AUG:-1}"

# EVAL3_EXTRA_REPOS is read by scripts/eval3_concat_patch.py; just propagate it.
export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-}"

echo ">> Eval3 aug-train"
echo "   dataset (primary)     : $REPO"
echo "   extras                : ${EVAL3_EXTRA_REPOS:-(none)}"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   max_frames_per_ep     : $EVAL3_MAX_FRAMES_PER_EP"
echo "   task_aug              : $EVAL3_TASK_AUG"
echo "   output dir            : $OUT"

# lerobot accepts --dataset.image_transforms.tfs only as a single JSON Dict
# (the deeper dotted paths like --dataset.image_transforms.tfs.brightness.weight
# are NOT recognised). So we pass the full tfs dict as one value.
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03]}}}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=2 \
  --rename_map="$RENAMES" \
  --dataset.repo_id="$REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs="$TFS_JSON" \
  --job_name="$JOB" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --batch_size="$BATCH" \
  "$@"
