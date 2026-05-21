#!/usr/bin/env bash
# Eval3 SmolVLA — v4 real slot corpora only (9 Hub datasets).
#
#   dataset_v4_{taylor,yann,barack}_{left,middle,right}
#
# Modes (first arg or EVAL3_V4SLOTS_MODE):
#   full    — full fine-tune from smolvla_base (batch 8)
#   expert  — freeze vision + train_expert_only (batch 16, bf16 on CUDA)
#
# Usage on Brev L40:
#   ./scripts/run_eval3_smolvla_v4slots_train.sh full
#   ./scripts/run_eval3_smolvla_v4slots_train.sh expert
#
# Detached (recommended):
#   EVAL3_TRAIN_CMD="./scripts/run_eval3_smolvla_v4slots_train.sh full" \
#   EVAL3_JOB_NAME=eval3-v4slots-full \
#     ./scripts/run_eval3_train_daemon.sh start
#
# Warm-start from v6 new66 (optional):
#   EVAL3_RESUME_FROM=RobotLearningVLA/eval3-vla-v6-smolvla-fresh-new66-50k \
#     ./scripts/run_eval3_smolvla_v4slots_train.sh full

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${EVAL3_V4SLOTS_MODE:-${1:-full}}"
shift || true

V4_PRIMARY="RobotLearningVLA/dataset_v4_taylor_left"
V4_EXTRAS="RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right,RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right"

export EVAL3_DATASET_REPO="${EVAL3_DATASET_REPO:-$V4_PRIMARY}"
export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-$V4_EXTRAS}"

export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
export EVAL3_TRAIN_STEPS="${EVAL3_TRAIN_STEPS:-50000}"
export EVAL3_SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"
export EVAL3_MAX_FRAMES_PER_EP="${EVAL3_MAX_FRAMES_PER_EP:-600}"

# Deploy-aligned task strings (data already canonical; aug keeps 100% canonical).
export EVAL3_TASK_AUG="${EVAL3_TASK_AUG:-1}"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-1.0}"

# v4 real: wide gripper already; no v2 gripper repair. Placement truncation ON
# via eval3_concat_patch (_is_new_data includes dataset_v4_* slot repos).
export EVAL3_GRIPPER_REPAIR="${EVAL3_GRIPPER_REPAIR:-0}"
export EVAL3_ACTION_SMOOTH_WINDOW="${EVAL3_ACTION_SMOOTH_WINDOW:-3}"
export EVAL3_ACTION_SMOOTH_GRIPPER="${EVAL3_ACTION_SMOOTH_GRIPPER:-0}"
export EVAL3_TRUNCATE_AT_PLACEMENT="${EVAL3_TRUNCATE_AT_PLACEMENT:-1}"

# Legacy print masks do not match v4 board geometry — keep OFF until slot masks exist.
export EVAL3_BG_REPLACE="${EVAL3_BG_REPLACE:-0}"
export EVAL3_PRINT_SHUFFLE="${EVAL3_PRINT_SHUFFLE:-0}"

# All episodes in these small corpora (no v1/v2 episode filters).
ALL_EPISODES="$(seq -s, 0 99)"
export EVAL3_SWIFT_EPISODE_FILTER="${EVAL3_SWIFT_EPISODE_FILTER:-$ALL_EPISODES}"
export EVAL3_LECUN_EPISODE_FILTER="${EVAL3_LECUN_EPISODE_FILTER:-$ALL_EPISODES}"
export EVAL3_OBAMA_EPISODE_FILTER="${EVAL3_OBAMA_EPISODE_FILTER:-$ALL_EPISODES}"

# Face-preserving aug (v10 Fix C) — celebrity prints stay readable.
DEFAULT_V10_TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.1,"p":0.3}},"resized_crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[480,640],"scale":[0.85,1.0],"ratio":[0.95,1.05]}},"gaussian_blur":{"weight":0.5,"type":"GaussianBlur","kwargs":{"kernel_size":[5,9],"sigma":[0.3,1.0]}},"erase":{"weight":0.25,"type":"RandomErasing","kwargs":{"p":0.15,"scale":[0.02,0.05]}}}'
export EVAL3_TFS_JSON="${EVAL3_TFS_JSON:-$DEFAULT_V10_TFS_JSON}"

export EVAL3_NUM_WORKERS="${EVAL3_NUM_WORKERS:-8}"

case "$MODE" in
  full)
    export EVAL3_BATCH="${EVAL3_BATCH:-8}"
    export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v6-smolvla-fresh-v4slots-50k}"
    export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-50k}"
    export EVAL3_FREEZE_VISION="${EVAL3_FREEZE_VISION:-0}"
    export EVAL3_TRAIN_EXPERT_ONLY="${EVAL3_TRAIN_EXPERT_ONLY:-0}"
    export EVAL3_USE_AMP="${EVAL3_USE_AMP:-0}"
    ;;
  expert)
    export EVAL3_BATCH="${EVAL3_BATCH:-16}"
    export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
    export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
    export EVAL3_FREEZE_VISION="${EVAL3_FREEZE_VISION:-1}"
    export EVAL3_TRAIN_EXPERT_ONLY="${EVAL3_TRAIN_EXPERT_ONLY:-1}"
    export EVAL3_USE_AMP="${EVAL3_USE_AMP:-1}"
    _BASE_BATCH=8
    _BASE_LR=1e-4
    if [[ -z "${EVAL3_PEAK_LR:-}" ]]; then
      export EVAL3_PEAK_LR="$(python3 -c "print(${_BASE_LR} * (${EVAL3_BATCH} / ${_BASE_BATCH}))")"
    fi
    ;;
  *)
    echo "Usage: $0 {full|expert}  (or EVAL3_V4SLOTS_MODE=full|expert)" >&2
    exit 2
    ;;
esac

echo ">> v4 slots SmolVLA train: mode=$MODE"
python3 tools/eval3_train_step_budget.py \
  --num-frames 33592 \
  --batch-size "$EVAL3_BATCH" \
  --target-epochs "${EVAL3_TARGET_EPOCHS:-30}" \
  || true

exec ./scripts/run_eval3_smolvla_aug_train.sh "$@"
