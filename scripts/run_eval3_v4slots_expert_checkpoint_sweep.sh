#!/usr/bin/env bash
# Offline sweep for the three deploy-battery v4slots-expert checkpoints:
#   v4slots_expert_30k  -> 030000
#   v4slots_expert_40k  -> 040000
#   v4slots_expert      -> 050000 (Hub snapshot symlinked if missing locally)
#
# Usage:
#   ./scripts/run_eval3_v4slots_expert_checkpoint_sweep.sh
#   EVAL3_SWEEP_DEVICE=mps ./scripts/run_eval3_v4slots_expert_checkpoint_sweep.sh
#
# Reports:
#   outputs/eval3_reports/v4slots_expert_30_40_50_sweep.{json,md}

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

TRAIN_DIR="outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k"
CKPT_50="${TRAIN_DIR}/checkpoints/050000/pretrained_model"
HUB_SNAP="${HOME}/.cache/huggingface/hub/models--RobotLearningVLA--eval3-vla-v6-smolvla-fresh-v4slots-expert-50k/snapshots"

if [[ ! -f "${CKPT_50}/model.safetensors" ]]; then
  if compgen -G "${HUB_SNAP}"/*/model.safetensors >/dev/null; then
    SNAP="$(ls -d "${HUB_SNAP}"/*/model.safetensors | head -1)"
    SNAP_DIR="$(dirname "$SNAP")"
    mkdir -p "${TRAIN_DIR}/checkpoints/050000"
    ln -sfn "$(cd "$SNAP_DIR" && pwd)" "${CKPT_50}"
    echo ">> Linked 050000 to Hub cache: ${SNAP_DIR}"
  else
    echo "ERROR: 050000 missing. Run deploy once (downloads Hub) or:" >&2
    echo "  bash scripts/fetch_eval3_v4slots_expert_checkpoints.sh 050000" >&2
    exit 2
  fi
fi

for step in 030000 040000; do
  if [[ ! -f "${TRAIN_DIR}/checkpoints/${step}/pretrained_model/model.safetensors" ]]; then
    echo "ERROR: missing ${step}. Run: bash scripts/fetch_eval3_v4slots_expert_checkpoints.sh ${step}" >&2
    exit 2
  fi
done

V4_REPOS="RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right,RobotLearningVLA/dataset_v4_taylor_left,RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right"

DEVICE="${EVAL3_SWEEP_DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  DEVICE="$(python - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
)"
fi

mkdir -p outputs/eval3_reports
echo ">> v4slots expert sweep (030000, 040000, 050000) device=${DEVICE}"

python scripts/eval3_smolvla_checkpoint_sweep.py \
  --train-dir "$TRAIN_DIR" \
  --checkpoints 030000,040000,050000 \
  --dataset-repos "$V4_REPOS" \
  --meta-repo-id RobotLearningVLA/dataset_v4_taylor_left \
  --rename-map '{"observation.images.front":"observation.images.camera1"}' \
  --device "$DEVICE" \
  --revision v3.0 \
  --video-backend pyav \
  --frames-per-episode 2 \
  --max-samples-per-repo 24 \
  --output-json outputs/eval3_reports/v4slots_expert_30_40_50_sweep.json \
  --output-md outputs/eval3_reports/v4slots_expert_30_40_50_sweep.md

echo ">> Wrote outputs/eval3_reports/v4slots_expert_30_40_50_sweep.md"
