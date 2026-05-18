#!/usr/bin/env bash
# Launch Eval3 OpenVLA LoRA exact-data runs with the requested OOM fallback.
#
# Examples:
#   OPENVLA_SRC=/workspace/openvla EVAL3_RECIPE=new66 integrations/openvla/scripts/run_eval3_lora_train.sh
#   OPENVLA_SRC=/workspace/openvla EVAL3_RECIPE=new_old88 integrations/openvla/scripts/run_eval3_lora_train.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ -d .venv_openvla_train ]]; then
  # shellcheck disable=SC1091
  source .venv_openvla_train/bin/activate
elif [[ -d .venv_openvla ]]; then
  # shellcheck disable=SC1091
  source .venv_openvla/bin/activate
fi

RECIPE="${EVAL3_RECIPE:-new66}"
case "$RECIPE" in
  new66)
    JOB="${EVAL3_JOB_NAME:-eval3-openvla-lora-new66-50k}"
    ;;
  new_old88)
    JOB="${EVAL3_JOB_NAME:-eval3-openvla-lora-new-old88-50k}"
    ;;
  *)
    echo "Unsupported EVAL3_RECIPE=$RECIPE; expected new66 or new_old88" >&2
    exit 2
    ;;
esac

OUT="${EVAL3_TRAIN_OUT:-outputs/train/$JOB}"
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
SAVE_STEPS="${EVAL3_SAVE_STEPS:-10000}"
BATCH="${EVAL3_BATCH:-4}"
GRAD_ACCUM="${EVAL3_GRAD_ACCUM:-4}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
OPENVLA_SRC="${OPENVLA_SRC:-}"
LOG_DIR="${OUT}/logs"
mkdir -p "$LOG_DIR"

run_train() {
  local batch="$1"
  shift
  local grad_accum="$1"
  shift
  local quantized="$1"
  shift
  local log_file="$LOG_DIR/train_b${batch}_ga${grad_accum}_q${quantized}.log"
  local cmd=(
    python integrations/openvla/scripts/train_eval3_lora.py
    --recipe "$RECIPE"
    --job-name "$JOB"
    --output-dir "$OUT"
    --openvla-src "$OPENVLA_SRC"
    --device "$DEVICE"
    --steps "$STEPS"
    --save-steps "$SAVE_STEPS"
    --batch-size "$batch"
    --grad-accumulation-steps "$grad_accum"
  )
  if [[ "$quantized" == "1" ]]; then
    cmd+=(--use-quantization)
  fi
  echo ">> ${cmd[*]}" | tee "$log_file"
  set +e
  "${cmd[@]}" "$@" 2>&1 | tee -a "$log_file"
  local status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

echo ">> Eval3 OpenVLA LoRA"
echo "   recipe        : $RECIPE"
echo "   job           : $JOB"
echo "   output        : $OUT"
echo "   steps         : $STEPS"
echo "   batch/accum   : $BATCH / $GRAD_ACCUM"
echo "   device        : $DEVICE"
echo "   openvla src   : ${OPENVLA_SRC:-(import from environment)}"
echo "   data labels   : exact; no gripper repair, no smoothing, no extra cap"

if run_train "$BATCH" "$GRAD_ACCUM" "0" "$@"; then
  exit 0
fi

if grep -Rqi "out of memory\\|cuda oom\\|CUDA out of memory" "$LOG_DIR"; then
  echo ">> CUDA OOM detected; retrying batch=2 grad_accum=8"
  if run_train "2" "8" "0" "$@"; then
    exit 0
  fi
  if grep -Rqi "out of memory\\|cuda oom\\|CUDA out of memory" "$LOG_DIR"; then
    echo ">> CUDA OOM persisted; retrying batch=2 grad_accum=8 with 4-bit quantization"
    run_train "2" "8" "1" "$@"
    exit $?
  fi
fi

echo "OpenVLA training failed; see $LOG_DIR" >&2
exit 1
