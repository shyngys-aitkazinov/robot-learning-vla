#!/usr/bin/env bash
# Generate the 9 pinned synthetic Eval3 datasets (5 target photos × K distractor
# combos per dataset = 5K episodes/dataset). Sources from
# dataset_v3_charuco_{left,middle,right}_1; outputs as
# dataset_v3_synth_pinned_<celeb>_<position>_1.
#
# Env vars (all optional):
#   EVAL3_SYNTH_PINNED_WORKERS         — multiprocessing workers (default 1; set 9 for full).
#   EVAL3_SYNTH_PINNED_PUSH_TO_HUB     — 0/1 (default 0; set 1 to upload + tag v3.0).
#   EVAL3_SYNTH_PINNED_K               — distractor combos per (celeb, photo). Default 10
#                                         (5×10=50 eps/dataset, 450 total). Max useful is 50
#                                         for ID pool (= full distractor grid).
#   EVAL3_SYNTH_PINNED_CELEBS          — comma list (default all 3).
#   EVAL3_SYNTH_PINNED_POSITIONS       — comma list (default all 3).
#   EVAL3_SYNTH_PINNED_VCODEC          — output mp4 codec (default h264).
#   EVAL3_SYNTH_PINNED_OVERWRITE       — 0/1 (default 0; set 1 to wipe existing outputs).
#   EVAL3_SYNTH_PINNED_CELEB_JSON      — defaults to ID-only datasets/in-distribution-eval-3.json.
#   EVAL3_SYNTH_PINNED_SEED            — distractor sampler seed (default 42).
#   EVAL3_SYNTH_PINNED_OUT_ROOT        — output dir root (default datasets).
#
# Usage:
#   ./scripts/run_eval3_synth_dataset_gen_pinned.sh                       # full local sweep
#   EVAL3_SYNTH_PINNED_WORKERS=9 EVAL3_SYNTH_PINNED_PUSH_TO_HUB=1 \
#     ./scripts/run_eval3_synth_dataset_gen_pinned.sh                      # Brev full + push
#   EVAL3_SYNTH_PINNED_CELEBS=taylor_swift EVAL3_SYNTH_PINNED_POSITIONS=left \
#     EVAL3_SYNTH_PINNED_K=2 \
#     ./scripts/run_eval3_synth_dataset_gen_pinned.sh                      # 10-ep smoke
#   ./scripts/run_eval3_synth_dataset_gen_pinned.sh --dry-run             # plan only

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS="${EVAL3_SYNTH_PINNED_WORKERS:-1}"
PUSH="${EVAL3_SYNTH_PINNED_PUSH_TO_HUB:-0}"
CELEBS="${EVAL3_SYNTH_PINNED_CELEBS:-taylor_swift,barack_obama,yann_lecun}"
POSITIONS="${EVAL3_SYNTH_PINNED_POSITIONS:-left,middle,right}"
K="${EVAL3_SYNTH_PINNED_K:-10}"
VCODEC="${EVAL3_SYNTH_PINNED_VCODEC:-h264}"
OVERWRITE="${EVAL3_SYNTH_PINNED_OVERWRITE:-0}"
CELEB_JSON="${EVAL3_SYNTH_PINNED_CELEB_JSON:-datasets/in-distribution-eval-3.json}"
SEED="${EVAL3_SYNTH_PINNED_SEED:-42}"
OUT_ROOT="${EVAL3_SYNTH_PINNED_OUT_ROOT:-datasets}"

ARGS=(
  --celebrity-json "$CELEB_JSON"
  --source-root datasets
  --out-root "$OUT_ROOT"
  --target-celebs "$CELEBS"
  --target-positions "$POSITIONS"
  --n-distractors-per-photo "$K"
  --distractor-seed "$SEED"
  --vcodec "$VCODEC"
  --n-workers "$WORKERS"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo ">> Eval3 pinned synth dataset gen"
echo "   workers       : $WORKERS"
echo "   push-to-hub   : $PUSH"
echo "   celebs        : $CELEBS"
echo "   positions     : $POSITIONS"
echo "   K (distract.) : $K"
echo "   vcodec        : $VCODEC"
echo "   overwrite     : $OVERWRITE"
echo "   celeb_json    : $CELEB_JSON"
echo "   distractor seed : $SEED"
echo "   out-root      : $OUT_ROOT"
echo ""

exec python tools/eval3_synth_dataset_gen_pinned.py "${ARGS[@]}" "$@"
