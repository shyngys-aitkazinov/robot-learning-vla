#!/usr/bin/env bash
# Eval3 SmolVLA fine-tune wrapper WITH the full data-prep + augmentation stack:
#   * Layer 1 (preprocessing): per-episode frame truncation to 600 frames (= 20s @ 30fps)
#     so the model only trains on 20s-budget-fittable trajectories. Driven by
#     env var EVAL3_MAX_FRAMES_PER_EP (default 600). 17/18 Obama episodes exceed 20s.
#   * Layer 2 (preprocessing): strict task-string augmentation at load time.
#     Driven by EVAL3_TASK_AUG (default 1). The recordings carry "Place the coke on
#     the <X>" but demo prompts will be "Place the coke on <X>". This layer mixes
#     80% canonical demo wording with 20% original recorded wording by default.
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
#     ./scripts/run_eval3_smolvla_aug_train.sh                                     # Brev

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="${EVAL3_DATASET_REPO:-RobotLearningVLA/taylor_swift_1}"
OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_3way_50k_v4_filtered_stats}"
JOB="${EVAL3_JOB_NAME:-eval3_3way_50k_v4_filtered_stats}"
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
BATCH="${EVAL3_BATCH:-8}"
DEVICE="${EVAL3_POLICY_DEVICE:-mps}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'

# Default: train from scratch (lerobot/smolvla_base). Set EVAL3_RESUME_FROM to a
# local checkpoint path to warm-restart instead.
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

# LR / scheduler. The previous 50k run wasted its last 20k steps at floor LR
# because num_decay_steps=30000 < total=50000. Now decay over (steps-warmup) so
# the entire run is productive.
PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-1000}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-49000}"
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"

# Save intermediate checkpoints so we can run synthetic-OOD across the run.

# Layers 1-4 prep are on by default; surface knobs for ablation runs.
export EVAL3_MAX_FRAMES_PER_EP="${EVAL3_MAX_FRAMES_PER_EP:-600}"
export EVAL3_TASK_AUG="${EVAL3_TASK_AUG:-1}"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.8}"
export EVAL3_BG_REPLACE="${EVAL3_BG_REPLACE:-1}"
export EVAL3_BG_REPLACE_P="${EVAL3_BG_REPLACE_P:-0.3}"
export EVAL3_PRINT_SHUFFLE="${EVAL3_PRINT_SHUFFLE:-1}"
export EVAL3_PRINT_SHUFFLE_P="${EVAL3_PRINT_SHUFFLE_P:-0.5}"
export EVAL3_MASK_DIR="${EVAL3_MASK_DIR:-outputs/eval3_masks}"
export EVAL3_BG_DIR="${EVAL3_BG_DIR:-outputs/eval3_backgrounds}"
# v6 plan: per-celebrity OLD-data filters target the NEGATIVE wrist_roll mode so
# the old data matches the NEW data's wrist_roll convention (~ -86 deg). The
# previous v3 filters kept the POSITIVE mode, which conflicts with the new data.
#   Swift OLD: keep audit-3 good episodes (unimodal wrist_roll, no flip needed).
#   LeCun OLD: keep only the 4 negative-mode episodes (matches new data).
#   Obama OLD: keep only the 4 negative-mode episodes (matches new data).
# Filters apply ONLY to old data; new dataset_v2_* uses all episodes (no filter).
export EVAL3_SWIFT_EPISODE_FILTER="${EVAL3_SWIFT_EPISODE_FILTER:-0,4,7,8,9,10,11,12,14,15,16,17,18,19}"
export EVAL3_LECUN_EPISODE_FILTER="${EVAL3_LECUN_EPISODE_FILTER:-3,7,9,13}"
export EVAL3_OBAMA_EPISODE_FILTER="${EVAL3_OBAMA_EPISODE_FILTER:-5,6,8,17}"

# Placement truncation: applied only to dataset_v2_* (new data records
# [home -> place -> home] cycles; truncate at first place event + 60-frame
# buffer so the trajectory ends at the placement pose like the old data).
export EVAL3_TRUNCATE_AT_PLACEMENT="${EVAL3_TRUNCATE_AT_PLACEMENT:-1}"
export EVAL3_TRUNCATE_GRIP_THRESHOLD="${EVAL3_TRUNCATE_GRIP_THRESHOLD:-20}"
export EVAL3_TRUNCATE_LIFT_THRESHOLD="${EVAL3_TRUNCATE_LIFT_THRESHOLD:--30}"
export EVAL3_TRUNCATE_BUFFER_FRAMES="${EVAL3_TRUNCATE_BUFFER_FRAMES:-60}"

# v6 default: ALL 11 repos (3 old + 8 dataset_v2_* minus the primary).
DEFAULT_EXTRA_REPOS="RobotLearningVLA/yann_lecun_1,RobotLearningVLA/barack_obama_1,RobotLearningVLA/dataset_v2_taylor_swift_left_1,RobotLearningVLA/dataset_v2_taylor_swift_middle_1,RobotLearningVLA/dataset_v2_taylor_swift_right_1,RobotLearningVLA/dataset_v2_yann_lecun_left_1,RobotLearningVLA/dataset_v2_yann_lecun_middle_1,RobotLearningVLA/dataset_v2_yann_lecun_right_1,RobotLearningVLA/dataset_v2_barack_obama_left_1,RobotLearningVLA/dataset_v2_barack_obama_middle_1,RobotLearningVLA/dataset_v2_barack_obama_right_1"
export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-$DEFAULT_EXTRA_REPOS}"

echo ">> Eval3 aug-train"
echo "   dataset (primary)     : $REPO"
echo "   extras                : ${EVAL3_EXTRA_REPOS:-(none)}"
echo "   policy.path           : $POLICY_PATH"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   peak_lr / decay_lr    : $PEAK_LR / $DECAY_LR"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   max_frames_per_ep     : $EVAL3_MAX_FRAMES_PER_EP"
echo "   task_aug              : $EVAL3_TASK_AUG"
echo "   task_aug canonical p  : $EVAL3_TASK_AUG_CANONICAL_P"
echo "   bg_replace (p)        : $EVAL3_BG_REPLACE ($EVAL3_BG_REPLACE_P)"
echo "   print_shuffle (p)     : $EVAL3_PRINT_SHUFFLE ($EVAL3_PRINT_SHUFFLE_P)"
echo "   truncate_at_placement : $EVAL3_TRUNCATE_AT_PLACEMENT (grip>=$EVAL3_TRUNCATE_GRIP_THRESHOLD, lift>=$EVAL3_TRUNCATE_LIFT_THRESHOLD, buf=$EVAL3_TRUNCATE_BUFFER_FRAMES)"
echo "   swift_episode_filter  : $EVAL3_SWIFT_EPISODE_FILTER"
echo "   lecun_episode_filter  : $EVAL3_LECUN_EPISODE_FILTER"
echo "   obama_episode_filter  : $EVAL3_OBAMA_EPISODE_FILTER"
echo "   save_freq             : $SAVE_FREQ"
echo "   output dir            : $OUT"

# lerobot accepts --dataset.image_transforms.tfs only as a single JSON Dict
# (the deeper dotted paths like --dataset.image_transforms.tfs.brightness.weight
# are NOT recognised). So we pass the full tfs dict as one value.
# The 4 new entries (perspective, resized_crop, gaussian_blur, erase) extend the
# existing 6 (brightness, contrast, saturation, hue, sharpness, affine).
# Bumped max_num_transforms 3 -> 4 to sample up to 4 transforms per step.
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03]}},"perspective":{"weight":1.5,"type":"RandomPerspective","kwargs":{"distortion_scale":0.2,"p":0.5}},"resized_crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[480,640],"scale":[0.75,1.0],"ratio":[0.95,1.05]}},"gaussian_blur":{"weight":0.5,"type":"GaussianBlur","kwargs":{"kernel_size":[5,9],"sigma":[0.3,2.0]}},"erase":{"weight":0.5,"type":"RandomErasing","kwargs":{"p":0.3,"scale":[0.02,0.1]}}}'

# IMPORTANT: lerobot's TrainPipelineConfig defaults use_policy_training_preset=true,
# which replaces our CLI optimizer/scheduler overrides with the SmolVLA preset
# (peak_lr=1e-4, num_decay_steps=30000). For the fresh-train plan we explicitly
# turn it OFF and supply optimizer + scheduler ourselves so the LR decays over
# the full 49k-step window instead of the first 30k.
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
  --dataset.image_transforms.max_num_transforms=4 \
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
