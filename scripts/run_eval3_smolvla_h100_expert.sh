#!/usr/bin/env bash
# Eval3 SmolVLA v10 — H100 expert-only recipe (patterns from TongxiHu/vla_eval1).
#
# Sets freeze_vision_encoder, train_expert_only, bf16, LR scaled with batch,
# and optional W&B + per-checkpoint Hub push.
#
# Usage (Brev H100):
#   EVAL3_POLICY_DEVICE=cuda ./scripts/run_eval3_smolvla_h100_expert.sh
#
# Smoke:
#   EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=8 ./scripts/run_eval3_smolvla_h100_expert.sh
#
# Optional Hub push each save_freq:
#   EVAL3_HUB_PUSH=1 EVAL3_HUB_REPO=RobotLearningVLA/eval3-smolvla-v10-balanced-h100-50k \
#     ./scripts/run_eval3_smolvla_h100_expert.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export EVAL3_V10_RECIPE="${EVAL3_V10_RECIPE:-v4_balanced_new66}"
export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
export EVAL3_BATCH="${EVAL3_BATCH:-16}"
export EVAL3_TRAIN_STEPS="${EVAL3_TRAIN_STEPS:-50000}"
export EVAL3_SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"
export EVAL3_NUM_WORKERS="${EVAL3_NUM_WORKERS:-12}"

export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v10-smolvla-fresh-v4balanced-new66-h100-expert-50k}"
export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v10-smolvla-fresh-v4balanced-new66-h100-expert-50k}"

# Expert-only + bf16 (vla_eval1 H100 defaults).
export EVAL3_FREEZE_VISION="${EVAL3_FREEZE_VISION:-1}"
export EVAL3_TRAIN_EXPERT_ONLY="${EVAL3_TRAIN_EXPERT_ONLY:-1}"
# bf16 via Accelerate when CUDA + use_amp (this lerobot build has no --policy.dtype).
export EVAL3_USE_AMP="${EVAL3_USE_AMP:-1}"

# Linear LR scaling: base 1e-4 @ batch 8 → scale with batch/8.
_BASE_BATCH=8
_BASE_LR=1e-4
if [[ -z "${EVAL3_PEAK_LR:-}" ]]; then
  export EVAL3_PEAK_LR="$(python3 -c "print(${_BASE_LR} * (${EVAL3_BATCH} / ${_BASE_BATCH}))")"
fi

echo ">> H100 expert v10 train"
python3 tools/eval3_train_step_budget.py \
  --recipe "$EVAL3_V10_RECIPE" \
  --batch-size "$EVAL3_BATCH" \
  --target-epochs "${EVAL3_TARGET_EPOCHS:-67}" \
  || true

exec ./scripts/run_eval3_smolvla_v10_train.sh "$@"
