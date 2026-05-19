#!/usr/bin/env bash
# Pins-pool synthetic LeRobot dataset generator wrapper.
#
# Produces dataset_v3_synth_<suffix>_<celeb>_<pos>_2 directories for every
# (target_celeb, target_position) combo in the chosen Pins pool. Each output
# is a self-contained LeRobot v3.0 dataset, ready to be referenced via
# EVAL3_EXTRA_REPOS in the SmolVLA training script.
#
# Defaults: top-30 pool, all 30 celebs as targets, all 3 positions, N=10
# target photos x M=50 distractor scenes = 500 configs per output dataset.
# 30 celebs x 3 positions = 90 output datasets per default run.
#
# Env vars (all optional):
#   EVAL3_PINS_WORKERS              — multiprocessing workers (default: nproc)
#   EVAL3_PINS_PUSH_TO_HUB          — 0/1 (default 0; set 1 to upload + tag v3.0)
#   EVAL3_PINS_POOL_JSON            — pool JSON (default datasets/pins-face-recognition-top30.json)
#   EVAL3_PINS_CELEBS               — comma list OR 'all' (default: all)
#   EVAL3_PINS_POSITIONS            — default: left,middle,right
#   EVAL3_PINS_MAX_PHOTOS           — N, photos per celeb (default: 10)
#   EVAL3_PINS_DISTRACTORS_PER_TARGET — M, distractor scenes per target_photo (default: 50)
#   EVAL3_PINS_OUTPUT_SUFFIX        — dataset name tag (default: pins30)
#   EVAL3_PINS_VCODEC               — h264 (default) or libsvtav1 for smaller files
#   EVAL3_PINS_OVERWRITE            — 0/1 (default 0)
#   EVAL3_PINS_SEED                 — RNG seed (default 42)
#
# Usage:
#   ./scripts/run_eval3_synth_pins_dataset_gen.sh                                    # full top-30 sweep
#   EVAL3_PINS_WORKERS=90 EVAL3_PINS_PUSH_TO_HUB=1 ./scripts/run_eval3_synth_pins_dataset_gen.sh
#   EVAL3_PINS_POOL_JSON=datasets/pins-face-recognition-top50.json EVAL3_PINS_OUTPUT_SUFFIX=pins50 \
#     ./scripts/run_eval3_synth_pins_dataset_gen.sh
#   ./scripts/run_eval3_synth_pins_dataset_gen.sh --dry-run                          # plan only

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

if command -v nproc >/dev/null 2>&1; then DEFAULT_W=$(nproc); else DEFAULT_W=$(sysctl -n hw.ncpu 2>/dev/null || echo 1); fi

WORKERS="${EVAL3_PINS_WORKERS:-$DEFAULT_W}"
PUSH="${EVAL3_PINS_PUSH_TO_HUB:-0}"
# Default to the QUALITY-FILTERED JSON when it exists, fall back to unfiltered.
# Re-generate with: python tools/pins_quality_filter.py --pool-json <raw> --out-json <quality>
_DEFAULT_POOL="datasets/pins-face-recognition-top30-quality.json"
if [[ ! -f "$_DEFAULT_POOL" ]]; then _DEFAULT_POOL="datasets/pins-face-recognition-top30.json"; fi
POOL_JSON="${EVAL3_PINS_POOL_JSON:-$_DEFAULT_POOL}"
CELEBS="${EVAL3_PINS_CELEBS:-all}"
POSITIONS="${EVAL3_PINS_POSITIONS:-left,middle,right}"
MAX_PHOTOS="${EVAL3_PINS_MAX_PHOTOS:-10}"
DISTRACTORS_PER_TARGET="${EVAL3_PINS_DISTRACTORS_PER_TARGET:-50}"
OUTPUT_SUFFIX="${EVAL3_PINS_OUTPUT_SUFFIX:-pins30}"
VCODEC="${EVAL3_PINS_VCODEC:-h264}"
OVERWRITE="${EVAL3_PINS_OVERWRITE:-0}"
SEED="${EVAL3_PINS_SEED:-42}"

ARGS=(
  --pool-json "$POOL_JSON"
  --target-celebs "$CELEBS"
  --target-positions "$POSITIONS"
  --max-photos-per-celeb "$MAX_PHOTOS"
  --distractors-per-target-photo "$DISTRACTORS_PER_TARGET"
  --output-suffix "$OUTPUT_SUFFIX"
  --vcodec "$VCODEC"
  --n-workers "$WORKERS"
  --seed "$SEED"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo ">> Eval 3 Pins-pool synth gen"
echo "   workers          : $WORKERS"
echo "   push-to-hub      : $PUSH"
echo "   pool             : $POOL_JSON"
echo "   celebs           : $CELEBS"
echo "   positions        : $POSITIONS"
echo "   N (max-photos)   : $MAX_PHOTOS"
echo "   M (distractors)  : $DISTRACTORS_PER_TARGET"
echo "   configs/ds       : $((MAX_PHOTOS * DISTRACTORS_PER_TARGET))"
echo "   output suffix    : $OUTPUT_SUFFIX"
echo "   vcodec           : $VCODEC"
echo "   overwrite        : $OVERWRITE"
echo ""

exec python tools/eval3_synth_pins_dataset_gen.py "${ARGS[@]}" "$@"
