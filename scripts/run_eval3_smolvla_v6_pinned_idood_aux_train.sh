#!/usr/bin/env bash
# Eval3 SmolVLA fine-tune on the pinned ID+OOD synth corpus WITH the full
# aux-head + state/language/visual augmentation stack.
#
# Trains on the 9 dataset_v3_synth_pinned_idood_<celeb>_<position>_2 datasets
# (K=35 pinned-distractor grid, ID+OOD 10-photo pool, ~3,150 episodes /
# 1.57M frames). The aux head + state augmentation break the
# (observation.state, image) -> action shortcut so the policy actually uses
# the language prompt. See docs/eval3/aux_head_playbook.md.
#
# Defaults match the agreed first-attempt recipe:
#   - warm start from eval3-smolvla-3way-25k-b128-v6-synth-step15k
#   - 5000 steps, batch 128, save_freq 500 (10 checkpoints)
#   - aux loss weight 0.5 (stronger than the playbook's 0.3 default)
#   - all state/language/visual augmentation at playbook defaults
#
# Smoke (~5 min):
#   EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=2 \
#     ./scripts/run_eval3_smolvla_v6_pinned_idood_aux_train.sh
#
# Full run:
#   ./scripts/run_eval3_smolvla_v6_pinned_idood_aux_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_3way_5k_b128_v6_pinned_idood_aux.log
#
# Knobs (all optional, defaults shown):
#   EVAL3_TRAIN_STEPS=5000    EVAL3_BATCH=128       EVAL3_POLICY_DEVICE=cuda
#   EVAL3_PEAK_LR=1e-4        EVAL3_DECAY_LR=1e-6   EVAL3_WARMUP_STEPS=250
#   EVAL3_DECAY_STEPS=4750    EVAL3_SAVE_FREQ=500
#   EVAL3_RESUME_FROM=RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k
#   EVAL3_AUX_POS_LOSS_WEIGHT=0.5   (sweep 0.2 / 0.3 / 0.5)
#   EVAL3_TRAIN_OUT=outputs/train/eval3_3way_5k_b128_v6_pinned_idood_aux
#   EVAL3_JOB_NAME=eval3_3way_5k_b128_v6_pinned_idood_aux
#   EVAL3_SKIP_PREFLIGHT=0    EVAL3_DRY_RUN=0    EVAL3_PREP_CACHE=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
EXTRA_REPOS="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_right_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_left_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_right_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_left_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_middle_2,RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_right_2"

OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3_3way_5k_b128_v6_pinned_idood_aux}"
JOB="${EVAL3_JOB_NAME:-eval3_3way_5k_b128_v6_pinned_idood_aux}"
STEPS="${EVAL3_TRAIN_STEPS:-5000}"
BATCH="${EVAL3_BATCH:-128}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
RENAMES='{"observation.images.front":"observation.images.camera1"}'
# Warm start from the v6-synth step15k snapshot (the aux_head_playbook's
# canonical starting point — cross-prompt baseline Δ ≈ 0.4°).
POLICY_PATH="${EVAL3_RESUME_FROM:-RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-250}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-4750}"
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-500}"

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
# Language augmentation — 70% canonical demo wording, 30% varied (incl.
# "image of X" phrasings). See docs/eval3/aux_head_playbook.md.
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.7}"
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"
export EVAL3_MASK_DIR="${EVAL3_MASK_DIR:-outputs/eval3_masks}"
export EVAL3_BG_DIR="${EVAL3_BG_DIR:-outputs/eval3_backgrounds}"

# Auxiliary position-classification head — default 0.5 here (stronger than
# the playbook's 0.3) per the agreed first-attempt recipe.
export EVAL3_AUX_POS_LOSS_WEIGHT="${EVAL3_AUX_POS_LOSS_WEIGHT:-0.5}"
export EVAL3_AUX_POS_DROPOUT="${EVAL3_AUX_POS_DROPOUT:-0.1}"
export EVAL3_AUX_POS_HIDDEN="${EVAL3_AUX_POS_HIDDEN:-256}"

# State augmentation — playbook defaults. Curriculum auto-spans $STEPS.
export EVAL3_STATE_NOISE_SIGMA_MAX="${EVAL3_STATE_NOISE_SIGMA_MAX:-0.3}"
export EVAL3_STATE_NOISE_SIGMA_MIN="${EVAL3_STATE_NOISE_SIGMA_MIN:-0.05}"
export EVAL3_STATE_NOISE_CURRICULUM_STEPS="${EVAL3_STATE_NOISE_CURRICULUM_STEPS:-${STEPS}}"
export EVAL3_STATE_REPLACE_PROB="${EVAL3_STATE_REPLACE_PROB:-0.4}"
export EVAL3_STATE_REPLACE_MODES="${EVAL3_STATE_REPLACE_MODES:-home:0.7,zero:0.3}"
export EVAL3_STATE_HOME_JITTER_SIGMA="${EVAL3_STATE_HOME_JITTER_SIGMA:-1.0}"
export EVAL3_STATE_GRIPPER_NOISE_SCALE="${EVAL3_STATE_GRIPPER_NOISE_SCALE:-0.1}"

unset EVAL3_SWIFT_EPISODE_FILTER
unset EVAL3_LECUN_EPISODE_FILTER
unset EVAL3_OBAMA_EPISODE_FILTER

echo ">> Eval3 v6 pinned ID+OOD aux-head fine-tune"
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   policy.path (warm)    : $POLICY_PATH"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   peak_lr / decay_lr    : $PEAK_LR / $DECAY_LR"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   task_aug canonical p  : $EVAL3_TASK_AUG_CANONICAL_P"
echo "   aux_pos_loss_weight   : $EVAL3_AUX_POS_LOSS_WEIGHT"
echo "   aux_pos_dropout       : $EVAL3_AUX_POS_DROPOUT"
echo "   aux_pos_hidden        : $EVAL3_AUX_POS_HIDDEN"
echo "   state_noise sigma     : $EVAL3_STATE_NOISE_SIGMA_MAX -> $EVAL3_STATE_NOISE_SIGMA_MIN over $EVAL3_STATE_NOISE_CURRICULUM_STEPS"
echo "   state_replace prob    : $EVAL3_STATE_REPLACE_PROB  modes=$EVAL3_STATE_REPLACE_MODES"
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

# Same image transforms as v6_synth: color/lighting, sharpness, small
# geometry, plus RandomErasing as a visual regularizer.
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
