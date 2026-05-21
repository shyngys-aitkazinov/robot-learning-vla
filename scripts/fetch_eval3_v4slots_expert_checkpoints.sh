#!/usr/bin/env bash
# Download v4slots-expert mid-training checkpoints from the Brev train instance.
# Hub only has step-050000; steps 010000–040000 live on eval3-v4slots-expert.
#
# Usage:
#   ./scripts/fetch_eval3_v4slots_expert_checkpoints.sh          # 10k–40k (~3.5 GB)
#   ./scripts/fetch_eval3_v4slots_expert_checkpoints.sh 050000   # include final too
#
# Env:
#   EVAL3_BREV_HOST   default: eval3-v4slots-expert
#   EVAL3_V4SLOTS_EXPERT_TRAIN_DIR  local train output root

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BREV_HOST="${EVAL3_BREV_HOST:-eval3-v4slots-expert}"
TRAIN_DIR="${EVAL3_V4SLOTS_EXPERT_TRAIN_DIR:-outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
REMOTE_CKPTS="/home/ubuntu/robot-learning-vla/${TRAIN_DIR}/checkpoints"
SSH_OPTS=(-o ControlMaster=no -o ControlPath=none)

STEPS=(010000 020000 030000 040000)
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
fi

echo ">> Fetching v4slots-expert checkpoints from ${BREV_HOST}"
echo "   remote: ${REMOTE_CKPTS}"
echo "   local : ${TRAIN_DIR}/checkpoints/"
echo ""

for step in "${STEPS[@]}"; do
  dest="${TRAIN_DIR}/checkpoints/${step}"
  if [[ -f "${dest}/pretrained_model/model.safetensors" ]]; then
    echo "   skip ${step} (already present)"
    continue
  fi
  mkdir -p "${dest}"
  echo "   tar ${step} (~866 MB) ..."
  # shadeform SSH cannot read ubuntu-owned train outputs; stream via sudo.
  ssh "${SSH_OPTS[@]}" "shadeform@${BREV_HOST}" \
    "sudo -u ubuntu tar cf - -C '${REMOTE_CKPTS}' '${step}'" \
    | tar xf - -C "${TRAIN_DIR}/checkpoints"
done

echo ""
echo ">> Done. Deploy with:"
echo "   FOLLOWER_TTY=/dev/tty.usbmodem5B140317761 ./scripts/run_eval3_deploy_battery.sh v4slots_expert_10k --task='Place the coke on Taylor Swift'"
