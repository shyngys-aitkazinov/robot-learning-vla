#!/usr/bin/env bash
# Generate the 9 synthetic ChArUco-based LeRobot training datasets.
#
# Defaults: full 2,250-episode sweep, serial (1 worker), local-only (no HF push).
# On a Brev cloud box: bump workers + enable upload.
#
# Env vars (all optional):
#   EVAL3_SYNTH_WORKERS       — multiprocessing workers (default 1; set 9 on Brev)
#   EVAL3_SYNTH_PUSH_TO_HUB   — 0/1 (default 0; set 1 to upload + tag v3.0)
#   EVAL3_SYNTH_CELEBS        — comma list (default taylor_swift,barack_obama,yann_lecun)
#   EVAL3_SYNTH_POSITIONS     — comma list (default left,middle,right)
#   EVAL3_SYNTH_N_CONFIGS     — episodes per dataset (default 250; lower for smoke)
#   EVAL3_SYNTH_VCODEC        — codec for output MP4s (default h264)
#   EVAL3_SYNTH_OVERWRITE     — 0/1 (default 0; set 1 to wipe existing outputs)
#   EVAL3_SYNTH_CELEB_JSON    — comma list (default datasets/in-distribution-eval-3.json).
#                                Add OOD by setting to
#                                'datasets/in-distribution-eval-3.json,datasets/out-distribution-eval-3.json'
#   EVAL3_SYNTH_OUTPUT_SUFFIX — tag appended to output names (default empty).
#                                Use 'ood' or 'mix' to write a parallel set of datasets
#                                alongside the ID-only ones.
#
# Usage:
#   ./scripts/run_eval3_synth_dataset_gen.sh                       # full local
#   EVAL3_SYNTH_WORKERS=9 EVAL3_SYNTH_PUSH_TO_HUB=1 \
#     ./scripts/run_eval3_synth_dataset_gen.sh                      # Brev runbook
#   EVAL3_SYNTH_CELEBS=taylor_swift EVAL3_SYNTH_POSITIONS=left \
#     EVAL3_SYNTH_N_CONFIGS=5 \
#     ./scripts/run_eval3_synth_dataset_gen.sh                      # 5-ep smoke
#   ./scripts/run_eval3_synth_dataset_gen.sh --dry-run             # plan only

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS="${EVAL3_SYNTH_WORKERS:-1}"
PUSH="${EVAL3_SYNTH_PUSH_TO_HUB:-0}"
CELEBS="${EVAL3_SYNTH_CELEBS:-taylor_swift,barack_obama,yann_lecun}"
POSITIONS="${EVAL3_SYNTH_POSITIONS:-left,middle,right}"
N_CONFIGS="${EVAL3_SYNTH_N_CONFIGS:--1}"  # -1 = full N*N*N*2 grid (250 for ID-only, 2000 for ID+OOD)
VCODEC="${EVAL3_SYNTH_VCODEC:-h264}"
OVERWRITE="${EVAL3_SYNTH_OVERWRITE:-0}"
CELEB_JSON="${EVAL3_SYNTH_CELEB_JSON:-datasets/in-distribution-eval-3.json}"
OUTPUT_SUFFIX="${EVAL3_SYNTH_OUTPUT_SUFFIX:-}"

ARGS=(
  --celebrity-json "$CELEB_JSON"
  --source-root datasets
  --out-root datasets
  --target-celebs "$CELEBS"
  --target-positions "$POSITIONS"
  --n-configs-per-dataset "$N_CONFIGS"
  --vcodec "$VCODEC"
  --n-workers "$WORKERS"
  --output-suffix "$OUTPUT_SUFFIX"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo ">> Eval 3 synth dataset gen"
echo "   workers       : $WORKERS"
echo "   push-to-hub   : $PUSH"
echo "   celebs        : $CELEBS"
echo "   positions     : $POSITIONS"
echo "   n_configs/ds  : $N_CONFIGS (-1 = full grid)"
echo "   vcodec        : $VCODEC"
echo "   overwrite     : $OVERWRITE"
echo "   celeb_jsons   : $CELEB_JSON"
echo "   output_suffix : '$OUTPUT_SUFFIX'"
echo ""

exec python tools/eval3_synth_dataset_gen.py "${ARGS[@]}" "$@"
