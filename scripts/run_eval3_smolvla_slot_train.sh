#!/usr/bin/env bash
# Eval3 SmolVLA — SLOT-BOTTLENECK fine-tune on the pinned ID+OOD synth corpus.
#
# Replaces the (leaky) suffix_out aux head with a prefix-side slot bottleneck:
#   * a SlotClassifier reads ONLY the image + language prefix tokens and emits
#     3-way slot logits (CE-supervised by target_position) + an h_slot vector;
#   * h_slot is appended as one extra prefix token the action expert attends to
#     (concat — raw language is kept);
#   * stop-grad on h_slot => the classifier is trained purely by the slot CE.
# See scripts/eval3_smolvla_slot_bottleneck.py.
#
# Paired with HEAVY state augmentation — and post-grasp states are extra-noisy
# (gripper-aperture heuristic) so the policy cannot ride proprio-continuation
# through the carry phase and must read the slot token to choose direction.
#
# Trains FRESH from lerobot/smolvla_base on the 9
# dataset_v3_synth_pinned_idood_<celeb>_<position>_2 datasets.
#
# An in-training eval watcher (tools/eval3_slot_eval_watcher.py) is backgrounded:
# every checkpoint it probes the 3 scenes with zeroed state under all 3 prompts
# and logs whether the model rotates to the correct slot.
#
# Smoke (~5 min):
#   EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=6 EVAL3_SLOT_EVAL_WATCH=0 \
#     ./scripts/run_eval3_smolvla_slot_train.sh
#
# Full run:
#   ./scripts/run_eval3_smolvla_slot_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_3way_5k_b128_slot.log
#
# Knobs (defaults shown):
#   EVAL3_TRAIN_STEPS=5000   EVAL3_BATCH=128      EVAL3_SAVE_FREQ=300
#   EVAL3_POLICY_DEVICE=cuda EVAL3_PEAK_LR=1e-4   EVAL3_DECAY_LR=1e-6
#   EVAL3_WARMUP_STEPS=250   EVAL3_DECAY_STEPS=4750
#   EVAL3_RESUME_FROM=lerobot/smolvla_base   (fresh-from-base by default)
#   EVAL3_SLOT_LOSS_WEIGHT=0.5  EVAL3_SLOT_STOPGRAD=1  EVAL3_SLOT_GUMBEL=0
#   EVAL3_SLOT_HIDDEN=256  EVAL3_SLOT_BOTTLENECK=64  EVAL3_SLOT_DROPOUT=0.1
#   EVAL3_SLOT_EVAL_WATCH=1  EVAL3_SKIP_PREFLIGHT=0  EVAL3_DRY_RUN=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
EXTRA_REPOS="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_right_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_left_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_right_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_left_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_right_2"

OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_3way_5k_b128_slot_bottleneck}"
JOB="${EVAL3_JOB_NAME:-eval3_3way_5k_b128_slot_bottleneck}"
STEPS="${EVAL3_TRAIN_STEPS:-5000}"
BATCH="${EVAL3_BATCH:-128}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-250}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-4750}"
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-300}"

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.7}"
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"

# --- Slot-bottleneck head (replaces the legacy aux head) ------------------
export EVAL3_SLOT_LOSS_WEIGHT="${EVAL3_SLOT_LOSS_WEIGHT:-0.5}"
export EVAL3_SLOT_HIDDEN="${EVAL3_SLOT_HIDDEN:-256}"
export EVAL3_SLOT_BOTTLENECK="${EVAL3_SLOT_BOTTLENECK:-64}"
export EVAL3_SLOT_DROPOUT="${EVAL3_SLOT_DROPOUT:-0.1}"
export EVAL3_SLOT_STOPGRAD="${EVAL3_SLOT_STOPGRAD:-1}"
export EVAL3_SLOT_GUMBEL="${EVAL3_SLOT_GUMBEL:-0}"
# Make sure the legacy aux head stays OFF (the entry script picks one).
unset EVAL3_AUX_POS_LOSS_WEIGHT

# --- HEAVY state augmentation + post-grasp amplification ------------------
# The slot bottleneck only fixes target *selection*; the state-continuation
# shortcut is broken here by heavily noising / replacing proprio, with the
# carry phase (gripper aperture < threshold) hit hardest.
export EVAL3_STATE_NOISE_SIGMA_MAX="${EVAL3_STATE_NOISE_SIGMA_MAX:-0.5}"
export EVAL3_STATE_NOISE_SIGMA_MIN="${EVAL3_STATE_NOISE_SIGMA_MIN:-0.1}"
export EVAL3_STATE_NOISE_CURRICULUM_STEPS="${EVAL3_STATE_NOISE_CURRICULUM_STEPS:-${STEPS}}"
export EVAL3_STATE_REPLACE_PROB="${EVAL3_STATE_REPLACE_PROB:-0.5}"
export EVAL3_STATE_REPLACE_MODES="${EVAL3_STATE_REPLACE_MODES:-home:0.6,zero:0.4}"
export EVAL3_STATE_HOME_JITTER_SIGMA="${EVAL3_STATE_HOME_JITTER_SIGMA:-1.0}"
export EVAL3_STATE_GRIPPER_NOISE_SCALE="${EVAL3_STATE_GRIPPER_NOISE_SCALE:-0.1}"
export EVAL3_STATE_POSTGRASP_GRIP_THRESH="${EVAL3_STATE_POSTGRASP_GRIP_THRESH:-35}"
export EVAL3_STATE_POSTGRASP_SIGMA_MULT="${EVAL3_STATE_POSTGRASP_SIGMA_MULT:-3.0}"
export EVAL3_STATE_POSTGRASP_REPLACE_PROB="${EVAL3_STATE_POSTGRASP_REPLACE_PROB:-0.9}"

unset EVAL3_SWIFT_EPISODE_FILTER
unset EVAL3_LECUN_EPISODE_FILTER
unset EVAL3_OBAMA_EPISODE_FILTER

echo ">> Eval3 SmolVLA slot-bottleneck fine-tune"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path           : $POLICY_PATH  (fresh-from-base)"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   save_freq             : $SAVE_FREQ"
echo "   slot loss weight      : $EVAL3_SLOT_LOSS_WEIGHT  (stopgrad=$EVAL3_SLOT_STOPGRAD gumbel=$EVAL3_SLOT_GUMBEL)"
echo "   slot hidden/bottleneck: $EVAL3_SLOT_HIDDEN / $EVAL3_SLOT_BOTTLENECK"
echo "   state noise sigma     : $EVAL3_STATE_NOISE_SIGMA_MAX -> $EVAL3_STATE_NOISE_SIGMA_MIN"
echo "   state replace prob    : $EVAL3_STATE_REPLACE_PROB  modes=$EVAL3_STATE_REPLACE_MODES"
echo "   post-grasp (grip<$EVAL3_STATE_POSTGRASP_GRIP_THRESH): sigma x$EVAL3_STATE_POSTGRASP_SIGMA_MULT, replace=$EVAL3_STATE_POSTGRASP_REPLACE_PROB"
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

# In-training eval watcher — backgrounded; survives the exec below.
if [[ "${EVAL3_SLOT_EVAL_WATCH:-1}" == "1" ]]; then
  mkdir -p outputs/train/logs
  WATCH_LOG="outputs/train/logs/${JOB}_watch.log"
  nohup python tools/eval3_slot_eval_watcher.py \
    --output-dir "$OUT" --device "$DEVICE" --final-step "$STEPS" \
    > "$WATCH_LOG" 2>&1 &
  echo ">> slot eval watcher backgrounded (pid $!) — log: $WATCH_LOG"
fi

TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.75,1.25]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"weight":0.5,"type":"ColorJitter","kwargs":{"hue":[-0.03,0.03]}},"sharpness":{"weight":0.75,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-2.0,2.0],"translate":[0.02,0.02]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.12,"p":0.3}},"random_erasing":{"weight":1.0,"type":"RandomErasing","kwargs":{"p":0.25,"scale":[0.02,0.15],"ratio":[0.3,3.3],"value":0}}}'

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
