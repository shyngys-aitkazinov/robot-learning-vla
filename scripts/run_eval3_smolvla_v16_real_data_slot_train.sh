#!/usr/bin/env bash
# Eval3 SmolVLA v16 — SLOT-BOTTLENECK fine-tune on the REAL dataset_v4_* corpus
# with the frame-0 / pre-grasp-CE fix.
#
# v16 vs v15 (run_eval3_smolvla_v15_real_data_slot_train.sh):
#   - The slot classifier reads the episode's FRAME-0 image, carried in the
#     camera2 slot (rename_map maps observation.images.front_frame0 ->
#     observation.images.camera2; empty_cameras drops 2 -> 1). The frame-0
#     scene is static (no coke motion) and counterfactual (same 3-face layout
#     appears with all 3 prompts) -> the slot CE loss can only drop by reading
#     the language prompt. h_slot is a function of camera2 = frame-0, so it is
#     automatically constant per episode (no freezing logic needed).
#   - The slot CE loss is counted ONLY on pre-grasp frames
#     (EVAL3_SLOT_CE_PREGRASP_ONLY=1) — the slot must be decided before the
#     carry begins. The gradient always propagates; there is no expert-freeze
#     warm-up phase.
#   - batch 128 (v15 used 256 — v16 embeds a second real camera).
#
# Both new behaviours are env-var-gated in eval3_dataset_prep.py and
# eval3_smolvla_slot_bottleneck.py; see docs/superpowers/plans/2026-05-20-v16-slot-bottleneck-fix.md.
#
# Smoke (~5 min, datasets cached from v15):
#   EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=12 EVAL3_WANDB=0 \
#     EVAL3_TRAIN_OUT=/ephemeral/outputs/train/eval3_v16_smoke \
#     EVAL3_JOB_NAME=eval3_v16_smoke \
#     ./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh --log_freq=2
#
# Full run:
#   ./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_v16_real_data_slot.log
#
# Knobs (defaults shown):
#   EVAL3_TRAIN_STEPS=10000  EVAL3_BATCH=128      EVAL3_SAVE_FREQ=1000
#   EVAL3_POLICY_DEVICE=cuda EVAL3_PEAK_LR=1e-4   EVAL3_DECAY_LR=1e-6
#   EVAL3_WARMUP_STEPS=250   EVAL3_DECAY_STEPS=(STEPS-WARMUP)
#   EVAL3_RESUME_FROM=lerobot/smolvla_base   (fresh-from-base)
#   EVAL3_SLOT_LOSS_WEIGHT=0.5  EVAL3_SLOT_HIDDEN=256  EVAL3_SLOT_BOTTLENECK=64
#   EVAL3_SLOT_DROPOUT=0.2  EVAL3_SLOT_STOPGRAD=1  EVAL3_SLOT_GUMBEL=0
#   EVAL3_SLOT_FRAME0=1  EVAL3_SLOT_CE_PREGRASP_ONLY=1  EVAL3_GRASP_GRIP_DELTA=20
#   EVAL3_WANDB=1  EVAL3_WANDB_PROJECT=eval3-v16-real-data-slot
#   EVAL3_DRY_RUN=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="RobotLearningVLA/dataset_v4_taylor_left"
EXTRA_REPOS="RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right,RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right"

OUT="${EVAL3_TRAIN_OUT:-/ephemeral/outputs/train/eval3_v16_real_data_slot}"
JOB="${EVAL3_JOB_NAME:-eval3_v16_real_data_slot}"
STEPS="${EVAL3_TRAIN_STEPS:-10000}"
BATCH="${EVAL3_BATCH:-128}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
# camera1 = current frame; camera2 = the episode's frame-0 scene (slot head).
RENAMES='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}'
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-250}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-$(( STEPS - WARMUP_STEPS ))}"
[[ "$DECAY_STEPS" -lt 1 ]] && DECAY_STEPS=1   # guard tiny-step smoke runs
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-1000}"

# draccus wants literal true/false for bool flags — map the 1/0 env knob.
if [[ "${EVAL3_WANDB:-1}" == "1" || "${EVAL3_WANDB:-1}" == "true" ]]; then
  WANDB_ENABLE="true"
else
  WANDB_ENABLE="false"
fi
WANDB_PROJECT="${EVAL3_WANDB_PROJECT:-eval3-v16-real-data-slot}"

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.7}"
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"
# v16 frame-0 mode is incompatible with the prep cache (the cache-hit path
# skips the frame-0 build) — keep the cache off.
export EVAL3_PREP_CACHE="0"

# --- Slot-bottleneck head + v16 frame-0 / pre-grasp-CE fix ----------------
export EVAL3_SLOT_LOSS_WEIGHT="${EVAL3_SLOT_LOSS_WEIGHT:-0.5}"
export EVAL3_SLOT_HIDDEN="${EVAL3_SLOT_HIDDEN:-256}"
export EVAL3_SLOT_BOTTLENECK="${EVAL3_SLOT_BOTTLENECK:-64}"
export EVAL3_SLOT_DROPOUT="${EVAL3_SLOT_DROPOUT:-0.2}"
export EVAL3_SLOT_STOPGRAD="${EVAL3_SLOT_STOPGRAD:-1}"
export EVAL3_SLOT_GUMBEL="${EVAL3_SLOT_GUMBEL:-0}"
export EVAL3_SLOT_FRAME0="${EVAL3_SLOT_FRAME0:-1}"
export EVAL3_SLOT_CE_PREGRASP_ONLY="${EVAL3_SLOT_CE_PREGRASP_ONLY:-1}"
export EVAL3_GRASP_GRIP_DELTA="${EVAL3_GRASP_GRIP_DELTA:-20}"
unset EVAL3_AUX_POS_LOSS_WEIGHT   # the entry script picks slot XOR aux

# --- Stochastic state augmentation — unchanged from v15 -------------------
export EVAL3_STATE_NOISE_SIGMA_MAX="${EVAL3_STATE_NOISE_SIGMA_MAX:-0.7}"
export EVAL3_STATE_NOISE_SIGMA_MIN="${EVAL3_STATE_NOISE_SIGMA_MIN:-0.15}"
export EVAL3_STATE_NOISE_CURRICULUM_STEPS="${EVAL3_STATE_NOISE_CURRICULUM_STEPS:-${STEPS}}"
export EVAL3_STATE_REPLACE_PROB="${EVAL3_STATE_REPLACE_PROB:-0.65}"
export EVAL3_STATE_REPLACE_MODES="${EVAL3_STATE_REPLACE_MODES:-home:0.5,zero:0.5}"
export EVAL3_STATE_HOME_JITTER_SIGMA="${EVAL3_STATE_HOME_JITTER_SIGMA:-1.5}"
export EVAL3_STATE_GRIPPER_NOISE_SCALE="${EVAL3_STATE_GRIPPER_NOISE_SCALE:-0.15}"
export EVAL3_STATE_POSTGRASP_GRIP_THRESH="${EVAL3_STATE_POSTGRASP_GRIP_THRESH:-35}"
export EVAL3_STATE_POSTGRASP_SIGMA_MULT="${EVAL3_STATE_POSTGRASP_SIGMA_MULT:-4.0}"
export EVAL3_STATE_POSTGRASP_REPLACE_PROB="${EVAL3_STATE_POSTGRASP_REPLACE_PROB:-0.95}"

unset EVAL3_SWIFT_EPISODE_FILTER
unset EVAL3_LECUN_EPISODE_FILTER
unset EVAL3_OBAMA_EPISODE_FILTER

echo ">> Eval3 v16 SmolVLA slot-bottleneck fine-tune — REAL dataset_v4_* + frame-0 fix"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path           : $POLICY_PATH  (fresh-from-base)"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   save_freq             : $SAVE_FREQ"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   wandb                 : enable=$WANDB_ENABLE project=$WANDB_PROJECT"
echo "   slot loss weight      : $EVAL3_SLOT_LOSS_WEIGHT  (frame0=$EVAL3_SLOT_FRAME0 ce_pregrasp_only=$EVAL3_SLOT_CE_PREGRASP_ONLY)"
echo "   grasp grip delta      : $EVAL3_GRASP_GRIP_DELTA"
echo "   rename_map            : $RENAMES"
echo "   output dir            : $OUT"

if [[ "${EVAL3_DRY_RUN:-0}" == "1" ]]; then
  echo ">> EVAL3_DRY_RUN=1; not starting training."
  exit 0
fi

# Same image transforms as v15.
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.5,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":0.75,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.04,0.04]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.18,"p":0.4}},"random_erasing":{"weight":1.5,"type":"RandomErasing","kwargs":{"p":0.4,"scale":[0.02,0.25],"ratio":[0.3,3.3],"value":0}}}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path="$POLICY_PATH" \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=true \
  --rename_map="$RENAMES" \
  --dataset.repo_id="$REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs="$TFS_JSON" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
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
