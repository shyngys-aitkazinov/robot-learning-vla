#!/usr/bin/env bash
# Course submission — Eval 3 in-distribution (Taylor Swift, Yann LeCun, Barack Obama).
#
# Installs deps if needed, then runs v4slots_expert (Hub checkpoint).
#
# Usage:
#   ./run_eval3_in_distribution.sh
#   ./run_eval3_in_distribution.sh "Place the coke on Yann LeCun"
#
# Env (override per machine):
#   FOLLOWER_TTY   — SO-101 serial (Mac: /dev/tty.usbmodem* ; Linux: /dev/ttyACM0)
#   CAM_IDX=0
#   EVAL3_POLICY_DEVICE — auto | cuda | mps | cpu
#   HF_TOKEN — required if Hub repos are private

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TASK="${1:-Place the coke on Taylor Swift}"

if [[ ! -d .venv ]]; then
  echo ">> First-time install (SmolVLA deps)…"
  EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export EVAL3_DEMO_POLICY="${EVAL3_DEMO_POLICY:-RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
export EVAL3_DEMO_PRESET="${EVAL3_DEMO_PRESET:-v4slots_expert}"
export FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/ttyUSB0}"
export CAM_IDX="${CAM_IDX:-0}"
export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-auto}"

echo ">> Eval 3 in-distribution — v4slots_expert"
echo "   policy : $EVAL3_DEMO_POLICY"
echo "   port   : $FOLLOWER_TTY   camera: $CAM_IDX   device: $EVAL3_POLICY_DEVICE"
echo "   task   : $TASK"
echo ""

exec ./scripts/run_eval3_demo_cli.sh --once "$TASK"
