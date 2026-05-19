#!/usr/bin/env bash
# Local-Mac shortcut for the Brev v10 pipeline status snapshot.
#
# Usage:
#   ./scripts/brev_v10_status.sh                # one-shot snapshot
#   ./scripts/brev_v10_status.sh --tail-orch    # also tail the orchestrator log
#   ./scripts/brev_v10_status.sh --tail-gen     # also tail the v4 gen log
#   ./scripts/brev_v10_status.sh --tail-train   # also tail the train log
#   ./scripts/brev_v10_status.sh --watch        # repeat every 60 s (Ctrl-C to stop)
#
# Bypasses macOS-incompatible ControlPath multiplexing automatically.
set -euo pipefail

HOST="${BREV_HOST:-brainy-tan-finch}"
SSH=(ssh -o ControlMaster=no -o ControlPath=none "$HOST")

snapshot() { "${SSH[@]}" "~/v10_status.sh"; }

case "${1:-}" in
  "")            snapshot ;;
  --tail-orch)   snapshot; echo; echo "--- orchestrator log (tail) ---"; "${SSH[@]}" 'tail -40 ~/robot-learning-vla/logs/v10_pipeline.log' ;;
  --tail-gen)    snapshot; echo; echo "--- v4 gen log (tail) ---";       "${SSH[@]}" 'tail -40 ~/robot-learning-vla/logs/v4_gen.log' ;;
  --tail-train)  snapshot; echo; echo "--- train log (tail) ---";        "${SSH[@]}" 'tail -40 ~/robot-learning-vla/logs/v10_train.log' ;;
  --watch)       while true; do clear; snapshot; sleep 60; done ;;
  *)             echo "unknown flag: $1" >&2; exit 2 ;;
esac
