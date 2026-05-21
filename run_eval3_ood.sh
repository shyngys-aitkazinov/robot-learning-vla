#!/usr/bin/env bash
# Course submission — Eval 3 OOD celebrities (not in v4slots training set).
#
# Uses teammate v16 slot-bottleneck checkpoint on Hub.
#
# Usage:
#   ./run_eval3_ood.sh "Place the coke on Lionel Messi"
#   ./run_eval3_ood.sh "Place the coke on Tom Cruise"
#   ./run_eval3_ood.sh "Place the coke on Leonardo DiCaprio"
#
# Env:
#   FOLLOWER_TTY, CAM_IDX, EVAL3_POLICY_DEVICE (cuda on Linux TA machine)
#   EVAL3_V16_CKPT — default Hub repo below
#   EVAL3_EPISODE_TIME_S=40  EVAL3_FPS=40  (friend recipe; override if needed)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TASK="${1:?Usage: $0 'Place the coke on <Celebrity Name>'}"

if [[ ! -d .venv ]]; then
  echo ">> First-time install (SmolVLA deps)…"
  EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export EVAL3_V16_CKPT="${EVAL3_V16_CKPT:-RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k}"
export FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/ttyUSB0}"
export CAM_IDX="${CAM_IDX:-0}"

# Device: prefer explicit env; else cuda on Linux, mps on Mac
if [[ -z "${EVAL3_POLICY_DEVICE:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export EVAL3_POLICY_DEVICE=mps
  else
    export EVAL3_POLICY_DEVICE=cuda
  fi
fi

EPISODE_S="${EVAL3_EPISODE_TIME_S:-40}"
FPS="${EVAL3_FPS:-40}"

echo ">> Eval 3 OOD — v16 pinsv5"
echo "   policy : $EVAL3_V16_CKPT"
echo "   port   : $FOLLOWER_TTY   camera: $CAM_IDX   device: $EVAL3_POLICY_DEVICE"
echo "   task   : $TASK"
echo "   timing : ${EPISODE_S}s @ ${FPS} Hz"
echo ""

exec ./scripts/run_eval3_deploy_battery.sh v16 \
  --task="$TASK" \
  --episode_time_s="$EPISODE_S" \
  --fps="$FPS" \
  --policy.device="$EVAL3_POLICY_DEVICE"
