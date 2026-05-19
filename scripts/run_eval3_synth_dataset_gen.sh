#!/usr/bin/env bash
# Generate the 9 synthetic ChArUco-based LeRobot training datasets.
#
# Defaults: full 2,250-episode sweep, serial (1 worker), local-only (no HF push).
# On a Brev cloud box: bump workers + enable upload.
#
# Env vars (all optional):
#   EVAL3_SYNTH_WORKERS              — multiprocessing workers (default 1).
#                                       Without sharding the upper limit is 9
#                                       (one per output dataset). With
#                                       --shards-per-dataset=N, set this to
#                                       N*9 to fully parallelize (e.g. 27 for
#                                       N=3 on the 28-core Brev box).
#   EVAL3_SYNTH_PUSH_TO_HUB          — 0/1 (default 0; set 1 to upload + tag v3.0).
#                                       With sharding only the MERGED dataset is pushed.
#   EVAL3_SYNTH_CELEBS               — comma list (default taylor_swift,barack_obama,yann_lecun)
#   EVAL3_SYNTH_POSITIONS            — comma list (default left,middle,right)
#   EVAL3_SYNTH_N_CONFIGS            — episodes per dataset (default 250; lower for smoke)
#   EVAL3_SYNTH_VCODEC               — codec for output MP4s (default h264)
#   EVAL3_SYNTH_OVERWRITE            — 0/1 (default 0; set 1 to wipe existing outputs)
#   EVAL3_SYNTH_CELEB_JSON           — comma list (default datasets/in-distribution-eval-3.json).
#                                       Add OOD by setting to
#                                       'datasets/in-distribution-eval-3.json,datasets/out-distribution-eval-3.json'
#   EVAL3_SYNTH_OUTPUT_SUFFIX        — tag appended to output names (default empty).
#                                       Use 'ood' or 'mix' to write a parallel set of datasets
#                                       alongside the ID-only ones.
#   EVAL3_SYNTH_BALANCED             — 0/1 (default 0). When 1, emit Fix A "language-forcing"
#                                       balanced datasets: ONE per celebrity (3 outputs total),
#                                       spanning all three placement slots, so the action label
#                                       varies WITHIN the dataset and the trainer cannot
#                                       shortcut on action = f(slot). Implies --target-positions
#                                       is ignored. See tools/eval3_diagnose_celeb_confusion.py
#                                       for the root-cause proof.
#   EVAL3_SYNTH_OUTPUT_VERSION       — v3 (default) or v4. Use v4 when --balanced=1 so the
#                                       outputs go to dataset_v4_synth_<celeb>_balanced_1.
#   EVAL3_SYNTH_SHARDS_PER_DATASET   — split each dataset into N shards built
#                                       in parallel, then merge via lerobot's
#                                       aggregate_datasets() (default 1 = no
#                                       sharding, byte-identical to legacy).
#                                       Set 2 or 3 to break the 9-worker cap.
#   EVAL3_SYNTH_KEEP_SHARDS          — 0/1 (default 0). With shards>1, set to 1
#                                       to preserve per-shard _shard_{i} dirs
#                                       after merge for debugging.
#
# Usage:
#   ./scripts/run_eval3_synth_dataset_gen.sh                       # full local
#   EVAL3_SYNTH_WORKERS=9 EVAL3_SYNTH_PUSH_TO_HUB=1 \
#     ./scripts/run_eval3_synth_dataset_gen.sh                      # Brev runbook (ID-only)
#   EVAL3_SYNTH_WORKERS=27 EVAL3_SYNTH_SHARDS_PER_DATASET=3 \
#     EVAL3_SYNTH_PUSH_TO_HUB=1 EVAL3_SYNTH_OUTPUT_SUFFIX=mix \
#     EVAL3_SYNTH_CELEB_JSON='datasets/in-distribution-eval-3.json,datasets/out-distribution-eval-3.json' \
#     ./scripts/run_eval3_synth_dataset_gen.sh                      # ID+OOD mix on 28-core box
#   EVAL3_SYNTH_CELEBS=taylor_swift EVAL3_SYNTH_POSITIONS=left \
#     EVAL3_SYNTH_N_CONFIGS=5 \
#     ./scripts/run_eval3_synth_dataset_gen.sh                      # 5-ep smoke
#   ./scripts/run_eval3_synth_dataset_gen.sh --dry-run             # plan only
#
#   # Fix A: generate the 3 v4 balanced datasets and push to Hub:
#   EVAL3_SYNTH_BALANCED=1 EVAL3_SYNTH_OUTPUT_VERSION=v4 \
#     EVAL3_SYNTH_WORKERS=3 EVAL3_SYNTH_PUSH_TO_HUB=1 \
#     ./scripts/run_eval3_synth_dataset_gen.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS="${EVAL3_SYNTH_WORKERS:-1}"
PUSH="${EVAL3_SYNTH_PUSH_TO_HUB:-0}"
CELEBS="${EVAL3_SYNTH_CELEBS:-taylor_swift,barack_obama,yann_lecun}"
POSITIONS="${EVAL3_SYNTH_POSITIONS:-left,middle,right}"
N_CONFIGS="${EVAL3_SYNTH_N_CONFIGS:--1}"  # -1 = full grid (250 for ID-only / 750 for --balanced)
VCODEC="${EVAL3_SYNTH_VCODEC:-h264}"
OVERWRITE="${EVAL3_SYNTH_OVERWRITE:-0}"
CELEB_JSON="${EVAL3_SYNTH_CELEB_JSON:-datasets/in-distribution-eval-3.json}"
OUTPUT_SUFFIX="${EVAL3_SYNTH_OUTPUT_SUFFIX:-}"
BALANCED="${EVAL3_SYNTH_BALANCED:-0}"
OUTPUT_VERSION="${EVAL3_SYNTH_OUTPUT_VERSION:-v3}"
SHARDS="${EVAL3_SYNTH_SHARDS_PER_DATASET:-1}"
KEEP_SHARDS="${EVAL3_SYNTH_KEEP_SHARDS:-0}"

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
  --output-version "$OUTPUT_VERSION"
  --shards-per-dataset "$SHARDS"
)
[[ "$PUSH" == "1" ]] && ARGS+=(--push-to-hub)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)
[[ "$BALANCED" == "1" ]] && ARGS+=(--balanced)
[[ "$KEEP_SHARDS" == "1" ]] && ARGS+=(--keep-shards)

echo ">> Eval 3 synth dataset gen"
echo "   workers          : $WORKERS"
echo "   push-to-hub      : $PUSH"
echo "   celebs           : $CELEBS"
echo "   positions        : $POSITIONS (ignored when balanced=1)"
echo "   n_configs/ds     : $N_CONFIGS (-1 = full grid)"
echo "   vcodec           : $VCODEC"
echo "   overwrite        : $OVERWRITE"
echo "   celeb_jsons      : $CELEB_JSON"
echo "   output_suffix    : '$OUTPUT_SUFFIX'"
echo "   output_version   : $OUTPUT_VERSION"
echo "   balanced         : $BALANCED  (Fix A: language-forcing layout)"
echo "   shards/dataset   : $SHARDS"
echo "   keep_shards      : $KEEP_SHARDS"
echo ""

exec python tools/eval3_synth_dataset_gen.py "${ARGS[@]}" "$@"
