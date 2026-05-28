#!/usr/bin/env bash
# Generate synthetic Eval3 datasets from the v5 charuko cross-product captures,
# warping PINS top-30 quality-filtered celebrities onto the boards.
#
# Sibling of run_eval3_synth_algvr_dataset_gen.sh: same v5 source family, same
# refactored generator (tools/eval3_synth_pins_dataset_gen.py), different pool
# (top-30 quality) and different scale (N=5 photos per celeb, M=3 distractor
# scenes per target photo by default).
#
# Sources: dataset_v5_charuko_{left,middle,right}_full (10 eps each, ~5k frames each)
# Pool   : datasets/pins-face-recognition-top30-quality.json (30 celebs, ~35 quality
#          photos each — quality_photos field sorted best-first by pins_quality_filter)
# Output : dataset_v5_synth_pins30q5_<celeb_slug>_<position>_full
#
# Per-dataset shape: N target photos x M distractor scenes per target photo.
#   Default N=5, M=3 -> 15 eps/dataset.
#   30 celebs x 3 positions = 90 datasets, ~1350 episodes, ~640k frames, ~3.4 GB.
#
# Env vars (all optional):
#   EVAL3_PINS30Q5_WORKERS       multiprocessing workers (default: cpu_count, capped at n_datasets)
#   EVAL3_PINS30Q5_M             distractor scenes per target photo (default: 3)
#   EVAL3_PINS30Q5_N             target photos per celeb (default: 5 — per user spec)
#   EVAL3_PINS30Q5_CELEBS        comma-separated slugs OR 'all' (default: all 30)
#   EVAL3_PINS30Q5_POSITIONS     comma list of {left,middle,right} (default: all 3)
#   EVAL3_PINS30Q5_PUSH_TO_HUB   0/1 (default: 0; set 1 to upload + tag v3.0)
#   EVAL3_PINS30Q5_OVERWRITE     0/1 (default: 0; set 1 to wipe existing outputs)
#   EVAL3_PINS30Q5_VCODEC        h264 (default) or libsvtav1 / libx265 / etc.
#   EVAL3_PINS30Q5_SEED          RNG seed for distractor sampling (default: 42)
#   EVAL3_PINS30Q5_POOL_JSON     defaults to datasets/pins-face-recognition-top30-quality.json
#   EVAL3_PINS30Q5_OUT_ROOT      output dir root (default: datasets)
#
# Usage:
#   ./scripts/run_eval3_synth_pins30q5_dataset_gen.sh                       # full sweep
#   ./scripts/run_eval3_synth_pins30q5_dataset_gen.sh --dry-run             # plan only
#   EVAL3_PINS30Q5_CELEBS=cristiano_ronaldo EVAL3_PINS30Q5_POSITIONS=left \
#     ./scripts/run_eval3_synth_pins30q5_dataset_gen.sh                      # 1-ds smoke

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS="${EVAL3_PINS30Q5_WORKERS:-0}"
M="${EVAL3_PINS30Q5_M:-3}"
N="${EVAL3_PINS30Q5_N:-5}"
CELEBS="${EVAL3_PINS30Q5_CELEBS:-all}"
POSITIONS="${EVAL3_PINS30Q5_POSITIONS:-left,middle,right}"
PUSH="${EVAL3_PINS30Q5_PUSH_TO_HUB:-0}"
OVERWRITE="${EVAL3_PINS30Q5_OVERWRITE:-0}"
VCODEC="${EVAL3_PINS30Q5_VCODEC:-h264}"
SEED="${EVAL3_PINS30Q5_SEED:-42}"
POOL_JSON="${EVAL3_PINS30Q5_POOL_JSON:-datasets/pins-face-recognition-top30-quality.json}"
OUT_ROOT="${EVAL3_PINS30Q5_OUT_ROOT:-datasets}"

ARGS=(
  --pool-json "$POOL_JSON"
  --source-root datasets
  --out-root "$OUT_ROOT"
  --target-celebs "$CELEBS"
  --target-positions "$POSITIONS"
  --max-photos-per-celeb "$N"
  --distractors-per-target-photo "$M"
  --vcodec "$VCODEC"
  --n-workers "$WORKERS"
  --seed "$SEED"
  --source-prefix "dataset_v5_charuko_"
  --source-suffix "_full"
  --output-prefix "dataset_v5_synth_"
  --output-postfix "_full"
  --output-suffix "pins30q5"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo ">> Eval3 PINS top-30-quality synth dataset gen (v5 charuko sources)"
echo "   pool          : $POOL_JSON"
echo "   workers       : $WORKERS"
echo "   N (target ph) : $N"
echo "   M (distract.) : $M"
echo "   celebs        : $CELEBS"
echo "   positions     : $POSITIONS"
echo "   push-to-hub   : $PUSH"
echo "   vcodec        : $VCODEC"
echo "   overwrite     : $OVERWRITE"
echo "   distractor seed : $SEED"
echo "   out-root      : $OUT_ROOT"
echo "   source        : dataset_v5_charuko_<pos>_full"
echo "   output        : dataset_v5_synth_pins30q5_<celeb>_<pos>_full"
echo ""

exec python tools/eval3_synth_pins_dataset_gen.py "${ARGS[@]}" "$@"
