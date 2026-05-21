#!/usr/bin/env bash
# Run on Brev instance after repo + scripts are present.
set -euo pipefail
MODE="${1:?Usage: remote_start_v4slots_train.sh full|expert}"
ROOT="${HOME}/robot-learning-vla"
cd "$ROOT"
source .venv/bin/activate
chmod +x scripts/run_eval3_smolvla_v4slots_train.sh scripts/run_eval3_train_daemon.sh 2>/dev/null || true

# Ensure install if startup did not finish
if [[ ! -d .venv ]]; then
  EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
fi

JOB="eval3-v4slots-${MODE}"
# Kill stale daemon if any
if [[ -f "logs/${JOB}.pid" ]]; then
  old="$(cat "logs/${JOB}.pid")"
  kill "$old" 2>/dev/null || true
  rm -f "logs/${JOB}.pid"
fi

export EVAL3_TRAIN_CMD="./scripts/run_eval3_smolvla_v4slots_train.sh ${MODE}"
export EVAL3_JOB_NAME="${JOB}"
./scripts/run_eval3_train_daemon.sh start
sleep 3
./scripts/run_eval3_train_daemon.sh status || true
echo "Log: ${ROOT}/logs/${JOB}.log"
tail -20 "logs/${JOB}.log" 2>/dev/null || true
