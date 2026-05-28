#!/usr/bin/env bash
# Generate synthetic Eval3 datasets from the v5 charuko cross-product captures,
# warping algvr.com conference celebrities onto the boards.
#
# Sources: dataset_v5_charuko_{left,middle,right}_full (10 eps each, ~5k frames each)
# Pool   : datasets/algvr-conference.json (34 organizers + invited speakers)
# Output : dataset_v5_synth_algvr_<celeb_slug>_<position>_full
#
# Per-dataset shape: N target_photos x M distractor scenes per target photo.
#   For algvr-conference (mostly 2 photos/celeb): default M=3 -> ~6 eps/dataset.
#   34 celebs x 3 positions = 102 datasets, ~600 episodes, ~300k frames, ~1.5 GB.
#
# Env vars (all optional):
#   EVAL3_ALGVR_WORKERS         multiprocessing workers (default: cpu_count, capped at n_datasets)
#   EVAL3_ALGVR_M               distractor scenes per target photo (default: 3)
#   EVAL3_ALGVR_N               cap on target photos per celeb (default: 4 -> uses all
#                                photos for every algvr celeb since max is 4 for LeCun)
#   EVAL3_ALGVR_CELEBS          comma-separated slugs OR 'all' (default: all 34)
#   EVAL3_ALGVR_POSITIONS       comma list of {left,middle,right} (default: all 3)
#   EVAL3_ALGVR_PUSH_TO_HUB     0/1 (default: 0; set 1 to upload + tag v3.0)
#   EVAL3_ALGVR_OVERWRITE       0/1 (default: 0; set 1 to wipe existing outputs)
#   EVAL3_ALGVR_VCODEC          h264 (default) or libsvtav1 / libx265 / etc.
#   EVAL3_ALGVR_SEED            RNG seed for distractor sampling (default: 42)
#   EVAL3_ALGVR_POOL_JSON       defaults to datasets/algvr-conference.json
#   EVAL3_ALGVR_OUT_ROOT        output dir root (default: datasets)
#
# Usage:
#   ./scripts/run_eval3_synth_algvr_dataset_gen.sh                       # full local sweep
#   ./scripts/run_eval3_synth_algvr_dataset_gen.sh --dry-run             # plan only
#   EVAL3_ALGVR_CELEBS=marc_pollefeys EVAL3_ALGVR_POSITIONS=left \
#     ./scripts/run_eval3_synth_algvr_dataset_gen.sh                      # 1-ds smoke

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS="${EVAL3_ALGVR_WORKERS:-0}"
M="${EVAL3_ALGVR_M:-3}"
N="${EVAL3_ALGVR_N:-4}"
CELEBS="${EVAL3_ALGVR_CELEBS:-all}"
POSITIONS="${EVAL3_ALGVR_POSITIONS:-left,middle,right}"
PUSH="${EVAL3_ALGVR_PUSH_TO_HUB:-0}"
OVERWRITE="${EVAL3_ALGVR_OVERWRITE:-0}"
VCODEC="${EVAL3_ALGVR_VCODEC:-h264}"
SEED="${EVAL3_ALGVR_SEED:-42}"
POOL_JSON="${EVAL3_ALGVR_POOL_JSON:-datasets/algvr-conference.json}"
OUT_ROOT="${EVAL3_ALGVR_OUT_ROOT:-datasets}"

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
  --output-suffix "algvr"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo ">> Eval3 algvr-conference synth dataset gen (v5 charuko sources)"
echo "   pool          : $POOL_JSON"
echo "   workers       : $WORKERS"
echo "   N (max photos): $N"
echo "   M (distract.) : $M"
echo "   celebs        : $CELEBS"
echo "   positions     : $POSITIONS"
echo "   push-to-hub   : $PUSH"
echo "   vcodec        : $VCODEC"
echo "   overwrite     : $OVERWRITE"
echo "   distractor seed : $SEED"
echo "   out-root      : $OUT_ROOT"
echo "   source        : dataset_v5_charuko_<pos>_full"
echo "   output        : dataset_v5_synth_algvr_<celeb>_<pos>_full"
echo ""

exec python tools/eval3_synth_pins_dataset_gen.py "${ARGS[@]}" "$@"
