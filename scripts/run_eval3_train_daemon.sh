#!/usr/bin/env bash
# Detached SmolVLA training launcher (vla_eval1-style PID / log / kill).
#
# Usage:
#   ./scripts/run_eval3_train_daemon.sh start
#   ./scripts/run_eval3_train_daemon.sh status
#   ./scripts/run_eval3_train_daemon.sh log
#   ./scripts/run_eval3_train_daemon.sh kill
#
# Override the underlying trainer:
#   EVAL3_TRAIN_CMD=./scripts/run_eval3_smolvla_h100_expert.sh ./scripts/run_eval3_train_daemon.sh start
#
# Job name (for log/pid filenames):
#   EVAL3_JOB_NAME=my-job ./scripts/run_eval3_train_daemon.sh start

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_CMD="${EVAL3_TRAIN_CMD:-./scripts/run_eval3_smolvla_v10_train.sh}"
JOB_NAME="${EVAL3_JOB_NAME:-eval3-smolvla-train}"
LOG_DIR="${EVAL3_LOG_DIR:-logs}"
PID_FILE="${LOG_DIR}/${JOB_NAME}.pid"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

mkdir -p "$LOG_DIR"

cmd="${1:-}"
shift || true

case "$cmd" in
  start)
    if [[ -f "$PID_FILE" ]]; then
      old_pid="$(cat "$PID_FILE")"
      if kill -0 "$old_pid" 2>/dev/null; then
        echo "Already running (PID $old_pid). Use: $0 kill"
        exit 1
      fi
      rm -f "$PID_FILE"
    fi
    if [[ -f "$LOG_FILE" ]]; then
      mv "$LOG_FILE" "${LOG_FILE}.bak"
    fi
    echo ">> Starting: $TRAIN_CMD $*"
    echo ">> Log: $LOG_FILE"
    nohup bash -lc "$TRAIN_CMD $*" >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Started PID $(cat "$PID_FILE")"
    else
      echo "Failed to start — tail $LOG_FILE"
      exit 1
    fi
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "RUNNING  PID=$(cat "$PID_FILE")  log=$LOG_FILE"
    else
      echo "NOT RUNNING (check $LOG_FILE)"
      exit 1
    fi
    ;;
  log)
    tail -f "$LOG_FILE"
    ;;
  kill)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "Stopped PID $pid"
      else
        echo "PID $pid not running"
      fi
      rm -f "$PID_FILE"
    else
      echo "No PID file: $PID_FILE"
    fi
    ;;
  *)
    echo "Usage: $0 {start|status|log|kill} [extra args passed to train script on start]" >&2
    exit 2
    ;;
esac
