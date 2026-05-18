#!/usr/bin/env bash
# Launch Eval3 FlowerVLA exact-data runs.
#
# Examples:
#   FLOWER_SRC=/workspace/flower_vla_calvin EVAL3_RECIPE=new66 ./scripts/run_eval3_flower_train.sh
#   FLOWER_SRC=/workspace/flower_vla_calvin EVAL3_RECIPE=new_old88 ./scripts/run_eval3_flower_train.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv_flower ]]; then
  # shellcheck disable=SC1091
  source .venv_flower/bin/activate
fi

RECIPE="${EVAL3_RECIPE:-new66}"
case "$RECIPE" in
  new66)
    JOB="${EVAL3_JOB_NAME:-eval3-flower-new66-50k}"
    ;;
  new_old88)
    JOB="${EVAL3_JOB_NAME:-eval3-flower-new-old88-50k}"
    ;;
  *)
    echo "Unsupported EVAL3_RECIPE=$RECIPE; expected new66 or new_old88" >&2
    exit 2
    ;;
esac

OUT="${EVAL3_TRAIN_OUT:-outputs/train/$JOB}"
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
BATCH="${EVAL3_BATCH:-8}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
FLOWER_SRC="${FLOWER_SRC:-}"

echo ">> Eval3 FlowerVLA"
echo "   recipe      : $RECIPE"
echo "   job         : $JOB"
echo "   output      : $OUT"
echo "   steps/batch : $STEPS / $BATCH"
echo "   device      : $DEVICE"
echo "   flower src  : ${FLOWER_SRC:-(import from environment)}"
echo "   data labels : exact; no gripper repair, no smoothing, no extra cap"

exec python scripts/train_eval3_flower.py \
  --recipe "$RECIPE" \
  --job-name "$JOB" \
  --output-dir "$OUT" \
  --flower-src "$FLOWER_SRC" \
  --device "$DEVICE" \
  --steps "$STEPS" \
  --batch-size "$BATCH" \
  --save-freq "$SAVE_FREQ" \
  "$@"
