#!/usr/bin/env bash
# Bootstrap + train-only pipeline for a fresh Brev H100 box.
# Skips synth gen — pulls v4 datasets from Hub at train time.
#
# Run on the H100 instance (after clone + HF token), or via:
#   brev shell <instance> -- bash -lc '~/robot-learning-vla/scripts/brev_h100_v10_train_only.sh'
#
# Env:
#   EVAL3_BATCH=32          default H100 batch
#   HUB_MODEL_REPO=...      Hub push target after train (default below)
#   SKIP_HUB_PUSH=1         train only, no hf upload

set -uo pipefail
cd "${HOME}/robot-learning-vla"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOG_DIR="logs"
mkdir -p "$LOG_DIR" outputs/eval3_diag
STATUS="$LOG_DIR/h100_v10_pipeline.status"
TRAIN_LOG="$LOG_DIR/h100_v10_train.log"
PUSH_LOG="$LOG_DIR/h100_v10_push.log"

JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v10-smolvla-fresh-v4balanced-new66-h100-50k}"
TRAIN_OUT="outputs/train/${JOB_NAME}"
HUB_MODEL_REPO="${HUB_MODEL_REPO:-RobotLearningVLA/eval3-smolvla-v10-balanced-h100-50k}"

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
stage() { echo "$(stamp) :: $*" | tee -a "$STATUS"; }

stage H100_PIPELINE_START

# Optional quick preflight (Hub parquets only, ~30s)
stage PREFLIGHT_START
set +e
python tools/eval3_diagnose_celeb_confusion.py \
  --mode v4_balanced \
  --celebs taylor_swift,yann_lecun,barack_obama \
  --out outputs/eval3_diag/v4_balanced_preflight_h100.json \
  > "$LOG_DIR/h100_v4_preflight.log" 2>&1
preflight_exit=$?
set -e
stage PREFLIGHT_EXIT=${preflight_exit}
if [ "$preflight_exit" -ge 2 ]; then
  stage PIPELINE_ABORT_PREFLIGHT_FAIL
  exit 2
fi

stage TRAIN_START_batch=${EVAL3_BATCH:-16}
set +e
EVAL3_POLICY_DEVICE=cuda \
EVAL3_BATCH="${EVAL3_BATCH:-16}" \
EVAL3_TRAIN_STEPS="${EVAL3_TRAIN_STEPS:-50000}" \
  ./scripts/run_eval3_smolvla_v10_h100_train.sh \
  > "$TRAIN_LOG" 2>&1
train_exit=$?
set -e
stage TRAIN_EXIT=${train_exit}
if [ "$train_exit" -ne 0 ]; then
  stage PIPELINE_TRAIN_FAILED
  exit 3
fi

if [ "${SKIP_HUB_PUSH:-0}" = "1" ]; then
  stage PIPELINE_COMPLETE_train_only
  exit 0
fi

stage PUSH_START_repo=${HUB_MODEL_REPO}
CKPT_DIR=""
for cand in "${TRAIN_OUT}/checkpoints/last/pretrained_model" \
            "${TRAIN_OUT}/checkpoints/050000/pretrained_model"; do
  if [ -d "$cand" ]; then
    CKPT_DIR="$cand"
    break
  fi
done
if [ -z "$CKPT_DIR" ]; then
  stage PUSH_FAIL_no_checkpoint
  exit 4
fi
stage PUSH_FOUND_CKPT=${CKPT_DIR}

set +e
hf upload "$HUB_MODEL_REPO" "$CKPT_DIR" \
  --repo-type model \
  --commit-message "v10 balanced+new66 50k H100 batch${EVAL3_BATCH:-32}" \
  > "$PUSH_LOG" 2>&1
push_exit=$?
set -e
stage PUSH_EXIT=${push_exit}
if [ "$push_exit" -ne 0 ]; then
  stage PIPELINE_PUSH_FAILED
  exit 5
fi

stage PIPELINE_COMPLETE_hub=${HUB_MODEL_REPO}
