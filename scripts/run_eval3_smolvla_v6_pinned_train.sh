#!/usr/bin/env bash
# Fine-tune SmolVLA on the pinned _1 synth datasets, warm-started from a v6
# synth checkpoint. Mirrors the v6_synth wrapper but points at the 9
# dataset_v3_synth_pinned_<celeb>_<position>_1 repos and defaults to the
# 25k-step15k checkpoint as the starting point + a short fine-tune budget.
#
# Defaults match the user's stated experiment:
#   - 2000 steps (short fine-tune)
#   - batch=128 (matches the v6 25k convention)
#   - save_freq=500 → 4 checkpoints (500, 1000, 1500, 2000)
#   - warmup=100, decay=1900 (cosine over the full run)
#   - peak_lr=1e-4 (conservative for fine-tune; the original 25k run hit ~1e-4
#     mid-decay around step 15k)
#   - AMP on (override at call time via --policy.use_amp=false if needed)
#
# Smoke (~5 min):
#   EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=2 \
#     ./scripts/run_eval3_smolvla_v6_pinned_train.sh
#
# Full Brev run:
#   tmux new -s eval3_v6_pinned
#   mkdir -p outputs/train/logs
#   ./scripts/run_eval3_smolvla_v6_pinned_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_3way_2k_b128_v6_pinned.log
#
# Knobs (all optional, defaults shown):
#   EVAL3_TRAIN_STEPS=2000      EVAL3_BATCH=128       EVAL3_POLICY_DEVICE=cuda
#   EVAL3_PEAK_LR=1e-4          EVAL3_DECAY_LR=1e-6   EVAL3_WARMUP_STEPS=100
#   EVAL3_DECAY_STEPS=1900      EVAL3_SAVE_FREQ=500
#   EVAL3_RESUME_FROM=RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k
#   EVAL3_TRAIN_OUT=outputs/train/eval3_3way_2k_b128_v6_pinned
#   EVAL3_JOB_NAME=eval3_3way_2k_b128_v6_pinned
#   EVAL3_SKIP_PREFLIGHT=0      EVAL3_DRY_RUN=0
#   EVAL3_PREP_CACHE=0 (opt-in)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# Pinned _1 datasets (produced by tools/eval3_synth_dataset_gen_pinned.py).
REPO="RobotLearningVLA/dataset_v3_synth_pinned_taylor_swift_left_1"
EXTRA_REPOS="RobotLearningVLA/dataset_v3_synth_pinned_taylor_swift_middle_1,RobotLearningVLA/dataset_v3_synth_pinned_taylor_swift_right_1,RobotLearningVLA/dataset_v3_synth_pinned_barack_obama_left_1,RobotLearningVLA/dataset_v3_synth_pinned_barack_obama_middle_1,RobotLearningVLA/dataset_v3_synth_pinned_barack_obama_right_1,RobotLearningVLA/dataset_v3_synth_pinned_yann_lecun_left_1,RobotLearningVLA/dataset_v3_synth_pinned_yann_lecun_middle_1,RobotLearningVLA/dataset_v3_synth_pinned_yann_lecun_right_1"

OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_3way_2k_b128_v6_pinned}"
JOB="${EVAL3_JOB_NAME:-eval3_3way_2k_b128_v6_pinned}"
STEPS="${EVAL3_TRAIN_STEPS:-2000}"
BATCH="${EVAL3_BATCH:-128}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'
# Warm-start by default from the step15k snapshot of the v6 25k run.
POLICY_PATH="${EVAL3_RESUME_FROM:-RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-100}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-1900}"
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-500}"

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

echo ">> Eval3 v6 pinned fine-tune train"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path (warm)    : $POLICY_PATH"
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
    --check-env \
    --no-strict-default-counts
fi

if [[ "${EVAL3_DRY_RUN:-0}" == "1" ]]; then
  echo ">> EVAL3_DRY_RUN=1; preflight passed, not starting training."
  exit 0
fi

# Same conservative image transforms as the v6_synth wrapper.
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
