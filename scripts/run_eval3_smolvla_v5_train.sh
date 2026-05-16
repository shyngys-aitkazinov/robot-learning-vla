#!/usr/bin/env bash
# Eval3 SmolVLA v5 launcher for the new 3-name x 3-position datasets.
#
# This intentionally does not reuse the v4 wrapper defaults. v5 trains only on
# the new datasets, disables legacy episode filters, disables the 600-frame cap,
# uses hard canonical prompt wording, and keeps image augmentation conservative.
#
# Smoke:
#   EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=2 EVAL3_POLICY_DEVICE=cuda ./scripts/run_eval3_smolvla_v5_train.sh
#
# Full Brev run:
#   tmux new -s eval3_v5
#   mkdir -p outputs/train/logs
#   EVAL3_POLICY_DEVICE=cuda ./scripts/run_eval3_smolvla_v5_train.sh 2>&1 | tee outputs/train/logs/eval3_3way_50k_v5_newdata_balanced.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="RobotLearningVLA/dataset_v2_barack_obama_left_1"
EXTRA_REPOS="RobotLearningVLA/dataset_v2_barack_obama_middle_1,RobotLearningVLA/dataset_v2_barack_obama_right_1,RobotLearningVLA/dataset_v2_yann_lecun_left_1,RobotLearningVLA/dataset_v2_yann_lecun_middle_1,RobotLearningVLA/dataset_v2_yann_lecun_middle_1,RobotLearningVLA/dataset_v2_yann_lecun_right_1,RobotLearningVLA/dataset_v2_yann_lecun_right_1,RobotLearningVLA/dataset_v2_taylor_swift_left_1,RobotLearningVLA/dataset_v2_taylor_swift_left_1,RobotLearningVLA/dataset_v2_taylor_swift_middle_1,RobotLearningVLA/dataset_v2_taylor_swift_middle_1,RobotLearningVLA/dataset_v2_taylor_swift_right_1,RobotLearningVLA/dataset_v2_taylor_swift_right_1"

OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_3way_50k_v5_newdata_balanced}"
JOB="${EVAL3_JOB_NAME:-eval3_3way_50k_v5_newdata_balanced}"
STEPS="${EVAL3_TRAIN_STEPS:-60000}"
BATCH="${EVAL3_BATCH:-8}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-1000}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-59000}"
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
export EVAL3_TASK_AUG_CANONICAL_P="1.0"
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"
export EVAL3_MASK_DIR="${EVAL3_MASK_DIR:-outputs/eval3_masks}"
export EVAL3_BG_DIR="${EVAL3_BG_DIR:-outputs/eval3_backgrounds}"

unset EVAL3_SWIFT_EPISODE_FILTER
unset EVAL3_LECUN_EPISODE_FILTER
unset EVAL3_OBAMA_EPISODE_FILTER

echo ">> Eval3 v5 new-data balanced train"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path           : $POLICY_PATH"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   peak_lr / decay_lr    : $PEAK_LR / $DECAY_LR"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   max_frames_per_ep     : $EVAL3_MAX_FRAMES_PER_EP"
echo "   task_aug canonical p  : $EVAL3_TASK_AUG_CANONICAL_P"
echo "   bg_replace            : $EVAL3_BG_REPLACE"
echo "   print_shuffle         : $EVAL3_PRINT_SHUFFLE"
echo "   save_freq             : $SAVE_FREQ"
echo "   output dir            : $OUT"

if [[ "${EVAL3_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python scripts/eval3_v5_dataset_preflight.py \
    --primary-repo "$REPO" \
    --extra-repos "$EVAL3_EXTRA_REPOS" \
    --check-env
fi

if [[ "${EVAL3_DRY_RUN:-0}" == "1" ]]; then
  echo ">> EVAL3_DRY_RUN=1; preflight passed, not starting training."
  exit 0
fi

# Conservative image transforms only: color/lighting, sharpness, small
# geometry changes. No random erase and no aggressive crop for v5 first run.
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.75,1.25]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"weight":0.5,"type":"ColorJitter","kwargs":{"hue":[-0.03,0.03]}},"sharpness":{"weight":0.75,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-2.0,2.0],"translate":[0.02,0.02]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.12,"p":0.3}}}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path="$POLICY_PATH" \
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
  --use_policy_training_preset=false \
  --optimizer.type=adamw \
  --optimizer.lr="$PEAK_LR" \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10.0 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.peak_lr="$PEAK_LR" \
  --scheduler.decay_lr="$DECAY_LR" \
  --scheduler.num_warmup_steps="$WARMUP_STEPS" \
  --scheduler.num_decay_steps="$DECAY_STEPS" \
  --job_name="$JOB" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --batch_size="$BATCH" \
  "$@"
