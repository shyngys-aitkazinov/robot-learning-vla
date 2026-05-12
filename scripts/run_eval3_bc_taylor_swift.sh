#!/usr/bin/env bash
# BC sanity training on the team's current Eval3-ish Hub dataset (default).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

REPO_ID="${EVAL3_REPO_ID:-RobotLearningVLA/taylor_swift_1}"
STEPS="${EVAL3_BC_STEPS:-2000}"
EPS="${EVAL3_EPISODES:-}" # e.g. export EVAL3_EPISODES="0" for single episode

ARGS=(
  python scripts/train_eval3_bc_overfit.py
  --repo-id "$REPO_ID"
  --steps "$STEPS"
  --video-backend pyav
  --output-dir outputs/eval3_bc_overfit
)

if [[ -n "${EVAL3_DEVICE:-}" ]]; then
  ARGS+=(--device "$EVAL3_DEVICE")
fi
if [[ -n "$EPS" ]]; then
  # shellcheck disable=SC2206
  EPS_ARR=($EPS)
  ARGS+=(--episodes "${EPS_ARR[@]}")
fi

exec "${ARGS[@]}"
