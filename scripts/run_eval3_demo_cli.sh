#!/usr/bin/env bash
# Eval 3 demo-day interactive CLI (Ubuntu TA workstation).
#
# Loads the policy once; the TA types celebrity prompts in a loop.
#
# Default model (v4slots expert — change anytime):
#   export EVAL3_DEMO_POLICY=RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k
#   ./scripts/run_eval3_demo_cli.sh
#
# Or pick a preset:
#   EVAL3_DEMO_PRESET=v4slots_expert ./scripts/run_eval3_demo_cli.sh
#   EVAL3_DEMO_PRESET=v16 EVAL3_DEMO_POLICY=RobotLearningVLA/eval3-vla-v16-real-synth-50k-step50k \
#     ./scripts/run_eval3_demo_cli.sh
#
# Hardware (Ubuntu defaults — override on the TA machine):
#   export FOLLOWER_TTY=/dev/ttyUSB0
#   export CAM_IDX=0
#   export EVAL3_POLICY_DEVICE=cuda
#
# Dry-run (no robot):
#   ./scripts/run_eval3_demo_cli.sh --dry-run
#
# Single rollout:
#   ./scripts/run_eval3_demo_cli.sh --once taylor

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Demo defaults — override via env before launching.
export EVAL3_DEMO_PRESET="${EVAL3_DEMO_PRESET:-v4slots_expert}"
export EVAL3_DEMO_POLICY="${EVAL3_DEMO_POLICY:-RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
export FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/ttyUSB0}"
export CAM_IDX="${CAM_IDX:-0}"
export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-auto}"
export EVAL3_EPISODE_TIME_S="${EVAL3_EPISODE_TIME_S:-20}"
export EVAL3_FPS="${EVAL3_FPS:-30}"

exec python scripts/eval3_demo_cli.py \
  --preset "$EVAL3_DEMO_PRESET" \
  --policy "$EVAL3_DEMO_POLICY" \
  --robot-port "$FOLLOWER_TTY" \
  --camera "$CAM_IDX" \
  --device "$EVAL3_POLICY_DEVICE" \
  --episode-time-s "$EVAL3_EPISODE_TIME_S" \
  --fps "$EVAL3_FPS" \
  "$@"
