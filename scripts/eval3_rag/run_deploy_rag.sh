#!/usr/bin/env bash
# Deploy launcher for the Eval3 RAG SmolVLA checkpoint.
#
# Wraps scripts/eval3_rag/deploy_rag.py with hardware defaults matching
# the Shyngys rig.  The deploy script injects the celebrity reference face
# (retrieved from EVAL3_RAG_REFDB) as camera2 so the model can visually
# match the reference face against the live scene.
#
# Prerequisites:
#   1.  EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
#   2.  python scripts/eval3_rag/build_celeb_refdb.py   # populate face DB
#   3.  A trained RAG checkpoint (run_train_rag.sh → outputs/train/eval3_rag/)
#
# Usage:
#   ./scripts/eval3_rag/run_deploy_rag.sh "Place the coke on Taylor Swift"
#   ./scripts/eval3_rag/run_deploy_rag.sh "Place the coke on Elon Musk"
#   ./scripts/eval3_rag/run_deploy_rag.sh "Place the coke on Lionel Messi"
#
# Env knobs (all have defaults):
#   EVAL3_RAG_CKPT        — path or Hub repo of RAG checkpoint
#   EVAL3_RAG_REFDB       — celebrity face DB directory
#   EVAL3_RAG_HW          — "H,W" load size for reference face
#   FOLLOWER_TTY          — follower arm serial port
#   CAM_IDX               — camera index
#   EVAL3_POLICY_DEVICE   — cuda / mps / cpu
#   EVAL3_EPISODE_TIME_S  — episode wall-clock budget (default 30)
#   EVAL3_FPS             — control loop frequency (default 30)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"

# ── keep alive after VSCode / SSH disconnect ──────────────────────────────────
if [[ -z "${TMUX:-}" && "${EVAL3_NO_TMUX:-0}" != "1" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    SESSION="eval3_deploy_$(date +%Y%m%d_%H%M%S)"
    LOG="$ROOT/outputs/logs/${SESSION}.log"
    mkdir -p "$ROOT/outputs/logs"
    _QSELF="$(printf '%q' "$_SELF")"
    _QLOG="$(printf '%q' "$LOG")"
    _QARGS=""
    for _a in "$@"; do _QARGS="${_QARGS} $(printf '%q' "$_a")"; done
    tmux new-session -d -s "$SESSION" \
      "EVAL3_NO_TMUX=1 bash ${_QSELF}${_QARGS} 2>&1 | tee ${_QLOG}"
    echo ">> Deploy launched in tmux session '$SESSION'"
    echo "   Log        : $LOG"
    echo "   Watch live : tail -f $LOG"
    echo "   Attach     : tmux attach -t $SESSION"
    exit 0
  else
    echo ">> WARNING: tmux not found — install: sudo apt-get install -y tmux"
    echo "   Running directly (job will die if VSCode/SSH disconnects)"
  fi
fi

TASK="${1:?Usage: $0 'Place the coke on <Celebrity Name>'}"

if [[ ! -d .venv ]]; then
  echo ">> First-time install (SmolVLA deps)..."
  EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── checkpoint ──────────────────────────────────────────────────────────────
CKPT="${EVAL3_RAG_CKPT:-outputs/train/test_eval3_v1/checkpoints/050000/pretrained_model}"
if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: RAG checkpoint not found at $CKPT" >&2
  echo "       Train first:  ./scripts/eval3_rag/run_train_rag.sh" >&2
  echo "       Or set:       EVAL3_RAG_CKPT=/path/to/pretrained_model" >&2
  exit 1
fi

# ── face DB ─────────────────────────────────────────────────────────────────
export EVAL3_RAG_REFDB="${EVAL3_RAG_REFDB:-datasets/celeb_refdb}"
export EVAL3_RAG_HW="${EVAL3_RAG_HW:-480,640}"

# ── task canonicalisation ────────────────────────────────────────────────────
# The model was trained with EVAL3_TASK_AUG_CANONICAL_P=1.0, so the task
# string NEVER contained a celebrity name during training.  At inference we
# extract the slug from the user-supplied task, load the reference image, then
# replace the task with this canonical string so the model sees the same
# distribution it was trained on.
# Override with EVAL3_CANONICAL_TASK= if your training used a different string.
export EVAL3_CANONICAL_TASK="${EVAL3_CANONICAL_TASK:-Place the coke on the person in the reference photo}"
# Set EVAL3_KEEP_TASK_NAME=1 to skip canonicalisation (ablation only).
export EVAL3_KEEP_TASK_NAME="${EVAL3_KEEP_TASK_NAME:-0}"
# Auto-fetch portraits from Wikipedia for anyone not already in EVAL3_RAG_REFDB.
# Enabled by default so unknown OOD celebrities work at demo day.
# Set EVAL3_RAG_AUTO_FETCH=0 for a fully-offline run.
export EVAL3_RAG_AUTO_FETCH="${EVAL3_RAG_AUTO_FETCH:-1}"

if [[ ! -d "$EVAL3_RAG_REFDB" ]]; then
  echo "ERROR: EVAL3_RAG_REFDB=$EVAL3_RAG_REFDB does not exist." >&2
  echo "       Run: python scripts/eval3_rag/build_celeb_refdb.py" >&2
  exit 1
fi

# ── hardware ─────────────────────────────────────────────────────────────────
export FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/ttyUSB0}"
export CAM_IDX="${CAM_IDX:-0}"

if [[ -z "${EVAL3_POLICY_DEVICE:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export EVAL3_POLICY_DEVICE=mps
  else
    export EVAL3_POLICY_DEVICE=cuda
  fi
fi

EPISODE_S="${EVAL3_EPISODE_TIME_S:-20}"
FPS="${EVAL3_FPS:-30}"

# ── rename map (same as training) ────────────────────────────────────────────
RENAMES='{"observation.images.front":"observation.images.camera1","observation.images.front_refface":"observation.images.camera2"}'

echo ">> Eval3 RAG SmolVLA deploy"
echo "   checkpoint : $CKPT"
echo "   refdb      : $EVAL3_RAG_REFDB  ($(ls "$EVAL3_RAG_REFDB" 2>/dev/null | wc -l) celebrity dirs)"
echo "   port       : $FOLLOWER_TTY   camera: $CAM_IDX   device: $EVAL3_POLICY_DEVICE"
echo "   task (user): $TASK"
if [[ "$EVAL3_KEEP_TASK_NAME" == "1" ]]; then
  echo "   task (model): $TASK  [EVAL3_KEEP_TASK_NAME=1, no canonicalisation]"
else
  echo "   task (model): $EVAL3_CANONICAL_TASK  [name stripped, reference photo injected]"
fi
echo "   timing     : ${EPISODE_S}s @ ${FPS} Hz"
echo ""

exec python scripts/eval3_rag/deploy_rag.py \
  --policy.path="$CKPT" \
  --policy.device="$EVAL3_POLICY_DEVICE" \
  --rename_map="$RENAMES" \
  --robot.port="$FOLLOWER_TTY" \
  --camera_index="$CAM_IDX" \
  --task="$TASK" \
  --episode_time_s="$EPISODE_S" \
  --fps="$FPS" \
  "$@"
