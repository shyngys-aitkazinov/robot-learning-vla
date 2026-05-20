#!/usr/bin/env bash
# Eval3 SmolVLA v15 — SLOT-BOTTLENECK fine-tune on the REAL dataset_v4_* corpus.
#
# This is the slot-bottleneck recipe (scripts/eval3_smolvla_slot_bottleneck.py)
# run for the FIRST time on REAL teleop recordings instead of synthetic
# ChArUco-composited data. The earlier slot run (v6_synth_pinned_idood) trained
# 100% on synthetic frames and failed both the language test (cross-prompt Δ
# ~2.4°) and on the real camera (domain gap). v15 removes the domain gap by
# training on real data and pushes regularization harder so the action expert
# cannot ride the (observation.state, image) -> action shortcut.
#
# Corpus: the 9 REAL datasets
#   RobotLearningVLA/dataset_v4_{taylor,barack,yann}_{left,middle,right}
# (~88 episodes / ~33.6k frames total — SMALL; heavy regularization is the
#  point, and the 1k-step checkpoints let you pick before overfit.)
#
# Differences vs run_eval3_smolvla_slot_train.sh:
#   - real dataset_v4_* corpus (not dataset_v3_synth_pinned_idood_*)
#   - 10000 steps, batch 256, save_freq 1000  (10 checkpoints)
#   - wandb enabled (project eval3-v15-real-data-slot)
#   - regularization turned UP across the board (state noise / replace /
#     post-grasp amplification / image transforms / slot dropout)
#   - vision encoder frozen explicitly (--policy.freeze_vision_encoder=true;
#     train_expert_only is already on in smolvla_base)
#   - preflight skipped (eval3_v5_dataset_preflight.py is hardcoded to the
#     dataset_v2 grid; v4 metadata was verified manually before this landed)
#   - output dir on /ephemeral (survived the repo outputs/ wipe; 573G free)
#
# Smoke (~5 min, downloads the 9 datasets):
#   EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=6 \
#     ./scripts/run_eval3_smolvla_v15_real_data_slot_train.sh
#
# Full run:
#   ./scripts/run_eval3_smolvla_v15_real_data_slot_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_v15_real_data_slot.log
#
# Knobs (defaults shown):
#   EVAL3_TRAIN_STEPS=10000  EVAL3_BATCH=256      EVAL3_SAVE_FREQ=1000
#   EVAL3_POLICY_DEVICE=cuda EVAL3_PEAK_LR=1e-4   EVAL3_DECAY_LR=1e-6
#   EVAL3_WARMUP_STEPS=250   EVAL3_DECAY_STEPS=(STEPS-WARMUP)
#   EVAL3_RESUME_FROM=lerobot/smolvla_base   (fresh-from-base)
#   EVAL3_SLOT_LOSS_WEIGHT=0.5  EVAL3_SLOT_STOPGRAD=1  EVAL3_SLOT_GUMBEL=0
#   EVAL3_SLOT_HIDDEN=256  EVAL3_SLOT_BOTTLENECK=64  EVAL3_SLOT_DROPOUT=0.2
#   EVAL3_WANDB=1  EVAL3_WANDB_PROJECT=eval3-v15-real-data-slot
#   EVAL3_SLOT_EVAL_WATCH=0  EVAL3_DRY_RUN=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="RobotLearningVLA/dataset_v4_taylor_left"
EXTRA_REPOS="RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right,RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right"

OUT="${EVAL3_TRAIN_OUT:-/ephemeral/outputs/train/eval3_v15_real_data_slot}"
JOB="${EVAL3_JOB_NAME:-eval3_v15_real_data_slot}"
STEPS="${EVAL3_TRAIN_STEPS:-10000}"
BATCH="${EVAL3_BATCH:-256}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'
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
WANDB_PROJECT="${EVAL3_WANDB_PROJECT:-eval3-v15-real-data-slot}"

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.7}"
# bg-replace / print-shuffle stay OFF: the extracted masks target legacy
# real-print geometry and dataset_v4_* uses short celeb slugs that the
# concat-patch slug matcher does not resolve anyway.
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"

# --- Slot-bottleneck head -------------------------------------------------
export EVAL3_SLOT_LOSS_WEIGHT="${EVAL3_SLOT_LOSS_WEIGHT:-0.5}"
export EVAL3_SLOT_HIDDEN="${EVAL3_SLOT_HIDDEN:-256}"
export EVAL3_SLOT_BOTTLENECK="${EVAL3_SLOT_BOTTLENECK:-64}"
export EVAL3_SLOT_DROPOUT="${EVAL3_SLOT_DROPOUT:-0.2}"   # up from 0.1
export EVAL3_SLOT_STOPGRAD="${EVAL3_SLOT_STOPGRAD:-1}"
export EVAL3_SLOT_GUMBEL="${EVAL3_SLOT_GUMBEL:-0}"
unset EVAL3_AUX_POS_LOSS_WEIGHT   # the entry script picks slot XOR aux

# --- HEAVY state augmentation (turned UP vs the slot launcher) ------------
# slot launcher was sigma 0.5->0.1, replace 0.5, post-grasp x3 / 0.9.
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

echo ">> Eval3 v15 SmolVLA slot-bottleneck fine-tune — REAL dataset_v4_* corpus"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path           : $POLICY_PATH  (fresh-from-base)"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   save_freq             : $SAVE_FREQ"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   wandb                 : enable=$WANDB_ENABLE project=$WANDB_PROJECT"
echo "   slot loss weight      : $EVAL3_SLOT_LOSS_WEIGHT  (stopgrad=$EVAL3_SLOT_STOPGRAD gumbel=$EVAL3_SLOT_GUMBEL dropout=$EVAL3_SLOT_DROPOUT)"
echo "   state noise sigma     : $EVAL3_STATE_NOISE_SIGMA_MAX -> $EVAL3_STATE_NOISE_SIGMA_MIN"
echo "   state replace prob    : $EVAL3_STATE_REPLACE_PROB  modes=$EVAL3_STATE_REPLACE_MODES"
echo "   post-grasp (grip<$EVAL3_STATE_POSTGRASP_GRIP_THRESH): sigma x$EVAL3_STATE_POSTGRASP_SIGMA_MULT, replace=$EVAL3_STATE_POSTGRASP_REPLACE_PROB"
echo "   output dir            : $OUT"

if [[ "${EVAL3_DRY_RUN:-0}" == "1" ]]; then
  echo ">> EVAL3_DRY_RUN=1; not starting training."
  exit 0
fi

# In-training slot eval watcher — OFF by default for v15: it probes synthetic
# scenes (domain-mismatched against a real-data model). wandb logs slot_loss /
# slot_acc directly. Set EVAL3_SLOT_EVAL_WATCH=1 to re-enable.
if [[ "${EVAL3_SLOT_EVAL_WATCH:-0}" == "1" ]]; then
  mkdir -p outputs/train/logs
  WATCH_LOG="outputs/train/logs/${JOB}_watch.log"
  nohup python tools/eval3_slot_eval_watcher.py \
    --output-dir "$OUT" --device "$DEVICE" --final-step "$STEPS" \
    > "$WATCH_LOG" 2>&1 &
  echo ">> slot eval watcher backgrounded (pid $!) — log: $WATCH_LOG"
fi

# Image transforms — turned UP vs the slot launcher (wider colour ranges,
# stronger perspective, RandomErasing p 0.25 -> 0.4 / scale to 0.25).
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.5,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":0.75,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.04,0.04]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.18,"p":0.4}},"random_erasing":{"weight":1.5,"type":"RandomErasing","kwargs":{"p":0.4,"scale":[0.02,0.25],"ratio":[0.3,3.3],"value":0}}}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path="$POLICY_PATH" \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=2 \
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
