#!/usr/bin/env bash
# Snapshot for the fast parallel v10 train on thin-amber-python (batch 32).
set -euo pipefail
HOST="${BREV_FAST_HOST:-thin-amber-python}"
ssh -o ControlMaster=no -o ControlPath=none "$HOST" '~/robot-learning-vla/logs/h100_v10_pipeline.status 2>/dev/null | tail -8; echo; tail -3 ~/robot-learning-vla/logs/h100_v10_train.log 2>/dev/null | tr "\r" "\n" | tail -4; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null'
