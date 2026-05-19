#!/usr/bin/env bash
# Eval3 SmolVLA v10 — H100 fast-train launcher (train-only, no synth gen).
#
# Same recipe as run_eval3_smolvla_v10_train.sh (v4_balanced_new66) but tuned
# for an 80 GiB H100: larger batch, distinct job/output names, and a separate
# Hub model repo so it does not collide with the L40S run.
#
# Target wall time: ~45–90 min for 50k steps on H100 (start batch=16;
# if step/sec is high after 200 steps, try EVAL3_BATCH=24; if OOM, use 12).
#
# Usage (on Brev H100 after install.sh + HF auth):
#   EVAL3_POLICY_DEVICE=cuda ./scripts/run_eval3_smolvla_v10_h100_train.sh
#
# Optional overrides:
#   EVAL3_BATCH=24          # if batch 32 OOMs
#   EVAL3_TRAIN_STEPS=50000
#   EVAL3_V10_RECIPE=v4_balanced_new66

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export EVAL3_V10_RECIPE="${EVAL3_V10_RECIPE:-v4_balanced_new66}"
export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
export EVAL3_BATCH="${EVAL3_BATCH:-16}"
export EVAL3_TRAIN_STEPS="${EVAL3_TRAIN_STEPS:-50000}"
export EVAL3_SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"

# Distinct from the L40S orchestrator run — avoids overwriting checkpoints.
export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v10-smolvla-fresh-v4balanced-new66-h100-50k}"
export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v10-smolvla-fresh-v4balanced-new66-h100-50k}"

echo ">> H100 fast v10 train"
echo "   recipe : $EVAL3_V10_RECIPE"
echo "   batch  : $EVAL3_BATCH"
echo "   steps  : $EVAL3_TRAIN_STEPS"
echo "   job    : $EVAL3_JOB_NAME"
echo "   out    : $EVAL3_TRAIN_OUT"

exec ./scripts/run_eval3_smolvla_v10_train.sh "$@"
